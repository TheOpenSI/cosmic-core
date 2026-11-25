### Core modules ###
import sys
import types
from pathlib import Path


### Type hint modules ###


### Internal modules ###


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

package = types.ModuleType("cosmic_core")
package.__path__ = [str(PROJECT_ROOT)]
sys.modules.setdefault("cosmic_core", package)