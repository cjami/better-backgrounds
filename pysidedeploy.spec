[app]
title = Better Backgrounds
project_dir = .
input_file = src\better_backgrounds\desktop\app.py
exec_directory = dist
project_file = pyproject.toml
icon = src/better_backgrounds/desktop/assets/app-icon.ico

[python]
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
macos.permissions = NSCameraUsageDescription:Better Backgrounds uses the selected camera to composite you locally into your room.
mode = standalone
extra_args = --quiet --zig --assume-yes-for-downloads --noinclude-qt-translations --output-filename=BetterBackgrounds --include-package=better_backgrounds --include-package=better_backgrounds._vendor.matanyone2 --include-package=better_backgrounds._vendor.pih --include-package=better_backgrounds._vendor.sharp --include-package=mediapipe --include-package=plyfile --include-package=pyvirtualcam --include-package=scipy --include-package=timm --include-package=torch --include-package=torchvision --include-package-data=better_backgrounds

[buildozer]
mode = debug
recipe_dir =
jars_dir =
ndk_path =
sdk_path =
local_libs =
arch =
