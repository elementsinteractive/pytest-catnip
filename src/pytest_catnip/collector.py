import logging
import types
from typing import override

import pytest
import yaml
from pydantic import ValidationError
from pytest_catnip.models import CatnipTestCaseData
from pytest_catnip.runner import create_test_function

logger = logging.getLogger(__name__)


class CatnipFileCollector(pytest.Module):
    @override
    def _getobj(self) -> types.ModuleType:
        """Generate a virtual Python module for this specific YAML file.

        Pytest will automatically crawl this module and let pytest-asyncio
        wire up your async execution loop and fixtures natively.
        """
        module_name = self.path.stem.replace("-", "_")
        mod = types.ModuleType(module_name)
        mod.__file__ = str(self.path)

        try:
            data = yaml.safe_load(self.path.read_text(encoding="utf-8"))
            if not data:
                return mod

            case_data = CatnipTestCaseData(**data, source_path=self.path)

            safe_name = case_data.name.replace("-", "_")
            func_name = safe_name if safe_name.startswith("test_") else f"test_{safe_name}"

            test_func = create_test_function(case_data)
            setattr(mod, func_name, test_func)

        except ValidationError as e:
            error_msgs = []
            for err in e.errors():
                loc_path = " -> ".join(str(loc) for loc in err["loc"])
                msg = err["msg"].replace("Extra inputs are not permitted", "Unknown field (check for typos)")
                error_msgs.append(f"  - [{loc_path}]: {msg}")

            formatted_errors = "\n".join(error_msgs)

            raise pytest.UsageError(f"Invalid test case schema in {self.path.name}:\n{formatted_errors}") from None

        except yaml.YAMLError as e:
            raise pytest.UsageError(f"Malformed YAML in {self.path.name}:\n{e}") from None

        except Exception as e:
            raise pytest.UsageError(f"Unexpected error loading test case from {self.path.name}: {e}") from e

        return mod
