"""Feature-first: Tests for progressive desktop startup presentation."""

from PySide6.QtWidgets import QApplication, QWidget

from better_backgrounds.desktop.startup import StartupCoordinator


def test_startup_coordinator_publishes_truthful_named_stages() -> None:
    """Show immediate named progress without inventing a percentage."""
    instance = QApplication.instance()
    application = instance if isinstance(instance, QApplication) else QApplication([])
    coordinator = StartupCoordinator(application)
    published = []
    coordinator.stage_changed.connect(published.append)
    window = QWidget()

    coordinator.show()
    coordinator.advance("preferences", "Loading preferences and rooms")
    coordinator.advance("matting", "Preparing background removal")
    window.show()
    coordinator.finish(window)

    assert [stage.key for stage in published] == [
        "starting",
        "preferences",
        "matting",
        "interactive",
    ]
    assert all(stage.elapsed_ms >= 0 for stage in published)
    window.close()
