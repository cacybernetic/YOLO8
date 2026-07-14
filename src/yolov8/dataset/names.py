"""Read the class names from a `data.yaml` file."""

import yaml


def parse_data_yaml(text):
    """Parse the content of a data.yaml file.

    The file must contain at least a `names` field with the class list.
    The `nc` field is optional; when present it must match `names`.

    Returns:
        List of class names, or None when the text has no `names` field.
    """
    if not text:
        return None
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        return None
    names = data.get('names')
    if names is None:
        return None
    if isinstance(names, dict):
        # Some tools export names as {0: 'door', 1: 'window', ...}.
        names = [names[k] for k in sorted(names)]
    names = [str(n) for n in names]
    nc = data.get('nc')
    if nc is not None and int(nc) != len(names):
        raise ValueError(
            f"data.yaml is not consistent: nc={nc} but "
            f"{len(names)} names are listed.")
    return names
