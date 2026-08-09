from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_generator_uses_the_vcstool_workspace_checkout():
    script = ROOT / "scripts" / "generate_morai_descriptors.py"
    spec = spec_from_file_location("generate_morai_descriptors", script)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    assert module.SOURCE == (
        ROOT.parent
        / "MORAI-DriveExample_GRPC"
        / "MoraiStandardProtos"
        / "Protos"
    )


def test_generated_descriptor_destination_is_the_dev_bridge_data_directory():
    script = ROOT / "scripts" / "generate_morai_descriptors.py"
    spec = spec_from_file_location("generate_morai_descriptors_output", script)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    assert module.OUTPUT == ROOT / "ad_morai_bridge_dev" / "data" / "morai_api.desc"
