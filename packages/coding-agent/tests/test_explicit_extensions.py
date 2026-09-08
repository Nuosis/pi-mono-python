import pytest
from pi_coding_agent.core.resource_loader import DefaultResourceLoader, DefaultResourceLoaderOptions

@pytest.mark.asyncio
async def test_explicit_extension_survives_disabled_discovery(tmp_path):
    explicit = tmp_path / "explicit.py"
    explicit.write_text('def extension_factory(pi):\n    pi.register_tool(name="probe", description="probe", parameters={"type":"object","properties":{}}, execute=lambda *a: None)\n')
    loader = DefaultResourceLoader(DefaultResourceLoaderOptions(cwd=str(tmp_path), agent_dir=str(tmp_path), additional_extension_paths=[str(explicit)], no_extensions=True))
    await loader.reload()
    result = loader.get_extensions()
    assert len(result["extensions"]) == 1, result
    assert "probe" in result["extensions"][0].tools

@pytest.mark.asyncio
async def test_disabled_discovery_without_explicit_extensions_is_empty(tmp_path):
    loader = DefaultResourceLoader(DefaultResourceLoaderOptions(cwd=str(tmp_path), agent_dir=str(tmp_path), no_extensions=True))
    await loader.reload()
    assert loader.get_extensions()["extensions"] == []
