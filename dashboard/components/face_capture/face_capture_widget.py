from pathlib import Path
import streamlit.components.v1 as components

_COMPONENT_DIR = Path(__file__).parent

# Declared inside a proper Python module so sys.modules contains this module name
face_capture_component = components.declare_component(
    "face_capture",
    path=str(_COMPONENT_DIR),
)
