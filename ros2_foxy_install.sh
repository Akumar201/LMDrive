#!/bin/bash
# ============================================================================
# ros2_foxy_install.sh
# Build ROS2 Foxy from source for Python 3.8 (conda env)
#
# Usage:
#   conda activate lmdrive
#   bash leaderboard/scripts/ros2_foxy_install.sh
#
# What this installs:
#   rclpy, std_msgs, sensor_msgs  (CycloneDDS as RMW)
#   Built for Python 3.8 inside the lmdrive conda environment
#   NOTE: cv_bridge is NOT needed — image conversion uses inline numpy instead
#
# Tested on: Ubuntu 22.04, conda env lmdrive (Python 3.8)
# ============================================================================

set -e

CONDA_PYTHON=/home/akumar/anaconda3/envs/lmdrive/bin/python3
WS=~/ros2_foxy_ws

# --------------------------------------------------------------------------
# 0. Preflight checks
# --------------------------------------------------------------------------
echo ""
echo "=== Preflight checks ==="

if ! command -v colcon &>/dev/null; then
  echo "Installing colcon..."
  pip install colcon-common-extensions colcon-cmake
fi

if ! command -v vcstool &>/dev/null && ! command -v vcs &>/dev/null; then
  echo "Installing vcstool..."
  pip install vcstool
fi

# empy 3.3.4 is REQUIRED — ROS2 Foxy uses em.BUFFERED_OPT which was removed in empy 4.x
EMPY_VER=$($CONDA_PYTHON -c "import em; print(em.__version__)" 2>/dev/null || echo "missing")
if [[ "$EMPY_VER" != "3.3.4" ]]; then
  echo "Downgrading empy to 3.3.4 (found: $EMPY_VER) ..."
  $CONDA_PYTHON -m pip install 'empy==3.3.4'
fi

# catkin_pkg is required by ament_cmake_core's package_xml_2_cmake.py
if ! $CONDA_PYTHON -c "import catkin_pkg" 2>/dev/null; then
  echo "Installing catkin_pkg..."
  $CONDA_PYTHON -m pip install catkin_pkg
fi

# cv_bridge is NOT used — image conversion is done with numpy directly in the agent files

# --------------------------------------------------------------------------
# 1. Fetch sources (skip if already done)
# --------------------------------------------------------------------------
echo ""
echo "=== Fetching ROS2 Foxy sources ==="

mkdir -p "$WS/src"
cd "$WS"

if [[ ! -f ros2.repos ]]; then
  curl -L https://raw.githubusercontent.com/ros2/ros2/foxy/ros2.repos -o ros2.repos
fi

if [[ $(ls src/ | wc -l) -lt 5 ]]; then
  vcs import src < ros2.repos
else
  echo "Sources already present, skipping vcs import."
fi

# --------------------------------------------------------------------------
# 2. Patch package.xml files  (all the fixes learned the hard way)
# --------------------------------------------------------------------------
echo ""
echo "=== Patching package.xml files ==="

# -- Fix 1: ament_cmake_google_benchmark -- its exec_depend on google_benchmark_vendor
#    causes colcon to require the vendor even though the cmake scripts don't need it.
patch_xml() {
  local FILE="$1"
  local OLD="$2"
  local NEW="$3"
  if grep -qF "$OLD" "$FILE" 2>/dev/null; then
    # Use Python for portable in-place string replace (avoids sed -i portability issues)
    python3 - "$FILE" "$OLD" "$NEW" <<'PYEOF'
import sys
f, old, new = sys.argv[1], sys.argv[2], sys.argv[3]
with open(f) as fh: content = fh.read()
with open(f, 'w') as fh: fh.write(content.replace(old, new, 1))
PYEOF
    echo "  Patched: $FILE"
  fi
}

patch_xml \
  "$WS/src/ament/ament_cmake/ament_cmake_google_benchmark/package.xml" \
  '<exec_depend>google_benchmark_vendor</exec_depend>' \
  '<!-- exec_depend google_benchmark_vendor disabled -->'

# -- Fix 2: rosidl_typesupport_fastrtps_c/cpp -- remove group membership so
#    rosidl_typesupport_c doesn't pull them in as group dependencies.
patch_xml \
  "$WS/src/ros2/rosidl_typesupport_fastrtps/rosidl_typesupport_fastrtps_c/package.xml" \
  '<member_of_group>rosidl_typesupport_c_packages</member_of_group>' \
  '<!-- member_of_group rosidl_typesupport_c_packages disabled -->'

patch_xml \
  "$WS/src/ros2/rosidl_typesupport_fastrtps/rosidl_typesupport_fastrtps_cpp/package.xml" \
  '<member_of_group>rosidl_typesupport_cpp_packages</member_of_group>' \
  '<!-- member_of_group rosidl_typesupport_cpp_packages disabled -->'

# -- Fix 3: rmw_fastrtps_cpp/dynamic_cpp -- remove group membership so
#    rmw_implementation does not require them via rmw_implementation_packages group.
patch_xml \
  "$WS/src/ros2/rmw_fastrtps/rmw_fastrtps_cpp/package.xml" \
  '<member_of_group>rmw_implementation_packages</member_of_group>' \
  '<!-- member_of_group rmw_implementation_packages disabled (rmw_fastrtps_cpp) -->'

patch_xml \
  "$WS/src/ros2/rmw_fastrtps/rmw_fastrtps_dynamic_cpp/package.xml" \
  '<member_of_group>rmw_implementation_packages</member_of_group>' \
  '<!-- member_of_group rmw_implementation_packages disabled (rmw_fastrtps_dynamic_cpp) -->'

# -- Fix 4: rmw_implementation -- remove build_depend on rmw_fastrtps_cpp
#    (only CycloneDDS will be used as the RMW).
patch_xml \
  "$WS/src/ros2/rmw_implementation/rmw_implementation/package.xml" \
  '<build_depend>rmw_fastrtps_cpp</build_depend>' \
  '<!-- build_depend rmw_fastrtps_cpp disabled -->'

# -- Fix 5: rosidl_generator_py -- test_depend on fastrtps_c is checked by colcon
#    even with BUILD_TESTING=OFF.
patch_xml \
  "$WS/src/ros2/rosidl_python/rosidl_generator_py/package.xml" \
  '<test_depend>rosidl_typesupport_fastrtps_c</test_depend>' \
  '<!-- test_depend rosidl_typesupport_fastrtps_c disabled -->'

# --------------------------------------------------------------------------
# 3. Pre-create stub install dirs for all skipped packages
#    Colcon checks for package.sh/local_setup.sh in install/ for every
#    declared dependency before invoking cmake.  Skipped packages need stubs.
# --------------------------------------------------------------------------
echo ""
echo "=== Creating stub install dirs for skipped packages ==="

SKIPPED_PKGS=(
  google_benchmark_vendor
  performance_test_fixture
  rcl_logging_log4cxx       # fails on Ubuntu 22.04: log4cxx uses std::shared_mutex (C++17 only)
  rosidl_typesupport_fastrtps_cpp
  rosidl_typesupport_fastrtps_c
  rmw_fastrtps_cpp
  rmw_fastrtps_dynamic_cpp
  rmw_fastrtps_shared_cpp
  fastrtps_cmake_module
)

for pkg in "${SKIPPED_PKGS[@]}"; do
  dir="$WS/install/${pkg}/share/${pkg}"
  mkdir -p "$dir"
  for ext in bash sh zsh; do
    printf '#!/bin/bash\n# stub — package was skipped at build time\n' \
      > "${dir}/package.${ext}"
    printf '#!/bin/bash\n# stub\n' \
      > "${dir}/local_setup.${ext}"
  done
  echo "  stub: $pkg"
done

# --------------------------------------------------------------------------
# 4. Build
# --------------------------------------------------------------------------
echo ""
echo "=== Building (this takes 2-5 minutes) ==="

unset AMENT_PREFIX_PATH
unset ROS_DISTRO

cd "$WS"
colcon build \
  --packages-up-to rclpy std_msgs sensor_msgs \
  --packages-skip \
    google_benchmark_vendor \
    performance_test_fixture \
    rcl_logging_log4cxx \
    rosidl_typesupport_fastrtps_cpp \
    rosidl_typesupport_fastrtps_c \
    rmw_fastrtps_cpp \
    rmw_fastrtps_dynamic_cpp \
    rmw_fastrtps_shared_cpp \
    fastrtps_cmake_module \
  --cmake-args \
    -DPYTHON_EXECUTABLE="$CONDA_PYTHON" \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DBUILD_TESTING=OFF \
    -DTRACETOOLS_DISABLED=ON \
    -DRMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
  --parallel-workers "$(nproc)"

# --------------------------------------------------------------------------
# 5. Verify
# --------------------------------------------------------------------------
echo ""
echo "=== Verifying ==="

source "$WS/install/setup.bash"

"$CONDA_PYTHON" - <<'PYEOF'
import sys
ok = True
for mod in ['rclpy', 'std_msgs.msg', 'sensor_msgs.msg']:
    try:
        __import__(mod)
        print(f"  OK  {mod}")
    except ImportError as e:
        print(f"  FAIL {mod}: {e}")
        ok = False
if ok:
    print("\nAll modules imported successfully.")
    print("Source the workspace with:")
    print("  source ~/ros2_foxy_ws/install/setup.bash")
else:
    print("\nSome imports failed — check errors above.")
    sys.exit(1)
PYEOF
