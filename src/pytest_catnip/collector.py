import logging
import types
from typing import override

import pytest
import yaml
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

        except Exception as e:
            raise pytest.UsageError(f"Error loading test case from {self.path}: {e}") from e

        return mod
