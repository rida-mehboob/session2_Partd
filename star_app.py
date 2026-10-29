"""UTS 36104 Session 2 Part D — Stars: interactive teaching model.
Inspired by the course's simplified mass-age-metallicity scaling relations.
"""
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

st.set_page_config(page_title="Stellar Life Explorer", page_icon="⭐", layout="wide")
SUN_T = 5772.0

# 1. Inputs: metallicity reruns Streamlit; mass and age control both charts.
st.title("⭐ Stellar Life Explorer")
st.caption("An illustrative stellar-evolution model • Mass, age and metallicity")
lz = st.slider("log10 metallicity", -4.0, -1.4, -1.7, step=0.05)
Z = 10 ** lz
zr = Z / 0.02
st.caption(f"Metallicity Z = {Z:.5f} (solar-relative z = {zr:.3f})")

with st.sidebar:
    st.header("Choose your star")
    log_mass = st.slider("log10 birth mass (solar masses)", -1.0, 2.45, 0.0, 0.05)
    log_age = st.slider("log10 age (billion years)", -4.0, 3.55, 0.65, 0.05)
    mass, age = 10 ** log_mass, 10 ** log_age
    st.metric("Birth mass", f"{mass:.2f} M☉")
    st.metric("Current age", f"{age:.3g} billion years")

def stage(m, z, age_gyr):
    """Simplified teaching model; not a stellar evolution simulation."""
    ratio = z / 0.02
    form = 0.03 * m ** -1.5
    ms_life = (10.0 * m ** -2.5 * (1 + 2.5 * np.exp(-m / 0.12)) + 0.0025) * ratio ** 0.1
    giant_end = 1.15 * ms_life
    basic_lum = m ** 3.5 * ratio ** -0.1
    basic_T = SUN_T * m ** 0.475 * ratio ** -0.025
    basic_R = np.sqrt(basic_lum) * (SUN_T / basic_T) ** 2

    if age_gyr < form:
        phase = "Protostar"
        T, L, R = max(2000, basic_T * 0.55), basic_lum * 0.2, basic_R * 3
    elif age_gyr <= ms_life:
        phase = "Main sequence"
        T, L, R = basic_T, basic_lum, basic_R
    elif m >= 0.33 and age_gyr <= giant_end:
        phase = "Giant / supergiant"
        progress = min(1.0, (age_gyr-ms_life)/max(giant_end-ms_life, 1e-9))
        T = max(2400, basic_T * (0.80-0.28*progress))
        L = basic_lum * (3+50*progress)
        R = np.sqrt(L)*(SUN_T/T)**2
    else:
        if m >= 140 and m <= 260 and z < 0.001:
            phase = "Pair-instability: no remnant"
            T, L, R = None, None, 0.0
        elif m >= 18 + 7 * ratio:
            phase = "Black hole"
            T, L, R = None, None, 0.0
        elif m >= 8:
            phase = "Neutron star"
            T, L, R = None, None, 0.0
        else:
            phase = "White dwarf"
            T = max(8500, min(40000, 22000 * (age_gyr/max(ms_life, 0.001))**-0.15))
            L = min(0.02, max(1e-5, 0.002*(T/22000)**4))
            R = np.sqrt(L)*(SUN_T/T)**2
    return {"phase": phase, "T": T, "L": L, "R": R,
            "ms_lifetime": ms_life, "black_hole_limit": 18+7*ratio}

@st.cache_data(show_spinner=False)
def make_phase_plane(z):
    # Fixed course-like phase plane. Logarithmically spaced mass and age.
    masses = np.geomspace(0.1, 282, 100)
    ages = np.geomspace(1e-4, 3500, 105)
    rows = []
    for m in masses:
        for a in ages:
            label = stage(float(m), z, float(a))["phase"]
            rows.append((float(m), float(a), label))
    return pd.DataFrame(rows, columns=["mass", "age", "phase"])

@st.cache_data(show_spinner=False)
def make_hr_band(z):
    ms = np.geomspace(0.12, 120, 240)
    zr = z / 0.02
    return pd.DataFrame({
        "temperature": SUN_T * ms ** 0.475 * zr ** -0.025,
        "luminosity": ms ** 3.5 * zr ** -0.1,
    })

state = stage(mass, Z, age)
left, right = st.columns([1,1], gap="large")

with left:
    st.subheader("Stellar mass–age phase plane")
    phases = ["Protostar", "Main sequence", "Giant / supergiant", "White dwarf",
              "Neutron star", "Black hole", "Pair-instability: no remnant"]
    palette = ["#7b8191", "#e3b65d", "#d76445", "#c9d9ee", "#8b7ed2", "#30313e", "#ffffff"]
    plane = make_phase_plane(Z)
    heat = alt.Chart(plane).mark_square(opacity=0.9, size=65).encode(
        x=alt.X("mass:Q", title="Birth mass (solar masses) — log scale",
                scale=alt.Scale(type="log", domain=[0.1, 282])),
        y=alt.Y("age:Q", title="Age (billion years) — log scale",
                scale=alt.Scale(type="log", domain=[1e-4, 3500])),
        color=alt.Color("phase:N", scale=alt.Scale(domain=phases, range=palette), title="Life stage"),
        tooltip=[alt.Tooltip("mass:Q", format=".2f"), alt.Tooltip("age:Q", format=".3g"), "phase:N"]
    )
    selected = pd.DataFrame([{"mass":mass, "age":age}])
    dot = alt.Chart(selected).mark_point(shape="diamond", filled=True, size=200,
                                          color="#00e5ff", stroke="#12252e", strokeWidth=1.3).encode(
        x="mass:Q", y="age:Q")
    st.altair_chart((heat + dot).properties(height=440).interactive(), use_container_width=True)

with right:
    st.subheader("Hertzsprung–Russell (HR) diagram")
    band = make_hr_band(Z)
    xscale = alt.Scale(type="log", domain=[2200, 45000], reverse=True, nice=False)
    yscale = alt.Scale(type="log", domain=[1e-4, 1e6], nice=False)
    main_sequence = alt.Chart(band).mark_line(color="#8b9da9", strokeWidth=12, opacity=0.55).encode(
        x=alt.X("temperature:Q", title="Surface temperature (K) — hot on the left", scale=xscale),
        y=alt.Y("luminosity:Q", title="Luminosity (solar units) — log scale", scale=yscale))
    hr = main_sequence
    if state["T"] is not None and state["L"] is not None and 2200 <= state["T"] <= 45000 and 1e-4 <= state["L"] <= 1e6:
        current = pd.DataFrame([{"temperature":state["T"], "luminosity":state["L"],"phase":state["phase"]}])
        point = alt.Chart(current).mark_point(size=290, filled=True, color="#eebd52",
                                               stroke="#2a3d4b", strokeWidth=2).encode(
            x="temperature:Q", y="luminosity:Q", tooltip=["phase:N", alt.Tooltip("temperature:Q",format=",.0f"),alt.Tooltip("luminosity:Q",format=".4g")])
        hr = hr + point
    st.altair_chart(hr.properties(height=440), use_container_width=True)
    if state["L"] is None:
        st.info("The selected phase does not appear on the HR diagram because it has no plotted stellar luminosity.")

st.divider()
st.subheader("Your star right now")
a,b,c,d = st.columns(4)
a.metric("Phase", state["phase"])
b.metric("Temperature", f"{state['T']:,.0f} K" if state["T"] is not None else "—")
c.metric("Luminosity", f"{state['L']:.3g} L☉" if state["L"] is not None else "—")
d.metric("Radius", f"{state['R']:.3g} R☉")
st.caption(f"Main-sequence lifetime: ~{state['ms_lifetime']:.3g} billion years. "
           f"Neutron-star/black-hole birth-mass boundary: {state['black_hole_limit']:.2f} M☉.")
st.caption("Course-style simplified model only; the life-stage boundaries and inferred physical values are illustrative.")
