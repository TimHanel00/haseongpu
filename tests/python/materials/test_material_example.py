from pathlib import Path
import sys


exampleDirectory = Path(__file__).resolve().parents[3] / "example"
sys.path.insert(0, str(exampleDirectory))

import materialApiExample  # noqa: E402


def test_material_example_builds_the_public_physical_graph():
    simulation = materialApiExample.buildSimulation()

    assert len(simulation.opticalComponents) == 2
    assert simulation.gainMedium.components == (simulation.opticalComponents[0],)
    material = simulation.gainMedium.components[0].material
    assert material.materialName == "Yb:YAG"
    assert material.isActive
    assert simulation.crossSections.resolution == material.crossSections.wavelengths.size
    assert simulation.opticalComponents[1].material.isPassive
    assert simulation.exteriorSurface.entityKind == "surface"
    assert len(simulation.opticalComponents[1].surfaceOptics) == 1
    assert simulation._backendGainMedium.get("claddingCellTypes").value.tolist() == [0, 1]
