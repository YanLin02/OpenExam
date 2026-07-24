# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata


def safe_copy_metadata(package_name):
    try:
        return copy_metadata(package_name)
    except Exception:
        return []


datas = [
    ("README.md", "."),
    ("openexam/app.py", "openexam"),
]
datas += collect_data_files("streamlit")

for package in (
    "streamlit",
    "altair",
    "numpy",
    "pandas",
    "pillow",
    "protobuf",
    "pyarrow",
    "requests",
    "tornado",
    "pydantic",
    "PyMuPDF",
    "python-docx",
    "python-pptx",
    "pywebview",
    "rapidfuzz",
):
    datas += safe_copy_metadata(package)

hiddenimports = []
hiddenimports += collect_submodules("openexam")
hiddenimports += collect_submodules("streamlit")
hiddenimports += collect_submodules("webview")

a = Analysis(
    ["openexam/macos_app.py"],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="OpenExam",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="OpenExam",
)
app = BUNDLE(
    coll,
    name="OpenExam.app",
    icon=None,
    bundle_identifier="dev.openexam.OpenExam",
    info_plist={
        "CFBundleName": "OpenExam",
        "CFBundleDisplayName": "OpenExam",
        "CFBundleShortVersionString": "0.1.0",
        "CFBundleVersion": "0.1.0",
        "LSMinimumSystemVersion": "10.15",
        "NSHighResolutionCapable": True,
    },
)
