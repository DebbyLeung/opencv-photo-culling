import os


def is_filtering_enabled(base_dir: str) -> bool:
    return not os.path.exists(os.path.join(base_dir, ".filtering-disabled"))


def set_filtering_enabled(base_dir: str, enabled: bool) -> None:
    disabled_marker = os.path.join(base_dir, ".filtering-disabled")
    if enabled:
        if os.path.exists(disabled_marker):
            os.remove(disabled_marker)
        return

    with open(disabled_marker, "w", encoding="utf-8"):
        pass
