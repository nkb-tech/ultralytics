from ultralytics.utils import NUMPY_VERSION
from ultralytics.utils.checks import check_version

NUMPY_2_0 = check_version(NUMPY_VERSION, ">=2.0.0")
