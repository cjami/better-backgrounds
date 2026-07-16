[app]
title = Better Backgrounds
project_dir = .
input_file = src\better_backgrounds\desktop\app.py
exec_directory = dist
project_file = pyproject.toml
icon = C:\Projects\better-backgrounds\.venv\Lib\site-packages\PySide6\scripts\deploy_lib\pyside_icon.ico

[python]
python_path = C:\Projects\better-backgrounds\.venv\Scripts\python.exe
packages = Nuitka==4.1.3

[qt]
qml_files =
excluded_qml_plugins = QtQuick,QtQuick3D,QtCharts,QtTest,QtSensors
modules = Core,Gui,Multimedia,WebChannel,WebEngineCore,WebEngineWidgets,Widgets
plugins = accessiblebridge,egldeviceintegrations,generic,iconengines,imageformats,multimedia,platforminputcontexts,platforms,platforms/darwin,platformthemes,styles,wayland-decoration-client,wayland-graphics-integration-client,wayland-shell-integration,xcbglintegrations

[android]
wheel_pyside =
wheel_shiboken =
plugins =

[nuitka]
macos.permissions =
mode = standalone
extra_args = --quiet --zig --assume-yes-for-downloads --noinclude-qt-translations --include-package=better_backgrounds --include-package-data=better_backgrounds

[buildozer]
mode = debug
recipe_dir =
jars_dir =
ndk_path =
sdk_path =
local_libs =
arch =
