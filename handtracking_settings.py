"""Immutable runtime settings and strict JSON loading."""

from dataclasses import dataclass
import json
import math
from pathlib import Path

from handtracking_config import MOVE_GAIN, POINTER_PINCH_OFF, POINTER_PINCH_ON


PROFILE_SCALES = {
    "standard": 1.0,
    "precisione": 0.65,
    "rapidita": 1.35,
}
PINCH_RELEASE_BRAKE_FRACTION = 0.6875
MAX_SETTINGS_BYTES = 64 * 1024
_SETTING_KEYS = frozenset(
    {"camera_index", "profile", "sensitivity", "pinch_on", "pinch_off"}
)


def _validate_profile(profile):
    if not isinstance(profile, str):
        raise ValueError("Tipo non valido per profile: attesa stringa")
    if profile not in PROFILE_SCALES:
        raise ValueError(f"Profilo non riconosciuto: {profile}")


def _validate_number(name, value, minimum, maximum):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Tipo non valido per {name}: atteso numero")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError(f"Valore non finito per {name}")
    if not minimum <= value <= maximum:
        raise ValueError(
            f"Valore fuori intervallo per {name}: "
            f"atteso tra {minimum} e {maximum}"
        )


def _validate_settings(
    camera_index, profile, sensitivity, pinch_on, pinch_off
):
    if isinstance(camera_index, bool) or not isinstance(camera_index, int):
        raise ValueError("Tipo non valido per camera_index: atteso intero")
    if not 0 <= camera_index <= 16:
        raise ValueError("Valore fuori intervallo per camera_index: atteso tra 0 e 16")
    _validate_profile(profile)
    _validate_number("sensitivity", sensitivity, 0.25, 3.0)
    _validate_number("pinch_on", pinch_on, 0.05, 1.5)
    _validate_number("pinch_off", pinch_off, 0.05, 1.5)
    if pinch_on >= pinch_off:
        raise ValueError("pinch_on deve essere minore di pinch_off")


@dataclass(frozen=True, slots=True)
class RuntimeSettings:
    camera_index: int = 0
    profile: str = "standard"
    sensitivity: float = 1.0
    pinch_on: float = POINTER_PINCH_ON
    pinch_off: float = POINTER_PINCH_OFF

    def __post_init__(self):
        _validate_settings(
            self.camera_index,
            self.profile,
            self.sensitivity,
            self.pinch_on,
            self.pinch_off,
        )
        object.__setattr__(self, "sensitivity", float(self.sensitivity))
        object.__setattr__(self, "pinch_on", float(self.pinch_on))
        object.__setattr__(self, "pinch_off", float(self.pinch_off))

    @property
    def move_gain(self):
        return MOVE_GAIN * self.sensitivity * PROFILE_SCALES[self.profile]

    @property
    def pinch_release_brake(self):
        return self.pinch_on + (
            self.pinch_off - self.pinch_on
        ) * PINCH_RELEASE_BRAKE_FRACTION


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Chiave JSON duplicata: {key}")
        result[key] = value
    return result


def _read_json(path):
    settings_path = Path(path)
    try:
        with settings_path.open("rb") as stream:
            raw = stream.read(MAX_SETTINGS_BYTES + 1)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"File impostazioni non trovato: {settings_path}"
        ) from exc
    except OSError as exc:
        raise ValueError(
            f"Impossibile leggere file impostazioni: {settings_path}"
        ) from exc
    if len(raw) > MAX_SETTINGS_BYTES:
        raise ValueError(
            f"File impostazioni troppo grande: massimo {MAX_SETTINGS_BYTES} byte"
        )
    try:
        text = raw.decode("utf-8-sig")
        data = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    except UnicodeDecodeError as exc:
        raise ValueError("File impostazioni non codificato in UTF-8") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON impostazioni non valido: {exc.msg}") from exc
    except RecursionError as exc:
        raise ValueError("JSON impostazioni troppo annidato") from exc
    if not isinstance(data, dict):
        raise ValueError("Le impostazioni devono essere un oggetto JSON")
    unknown = sorted(set(data) - _SETTING_KEYS)
    if unknown:
        raise ValueError(
            "Chiavi impostazioni non riconosciute: " + ", ".join(unknown)
        )
    return data


def load_settings(path=None, profile=None):
    """Load optional JSON overrides, or return defaults when path is absent."""
    data = {} if path is None else _read_json(path)
    if "profile" in data:
        _validate_profile(data["profile"])
    if profile is not None:
        _validate_profile(profile)
    values = {
        "camera_index": 0,
        "profile": "standard",
        "sensitivity": 1.0,
        "pinch_on": POINTER_PINCH_ON,
        "pinch_off": POINTER_PINCH_OFF,
    }
    values.update(data)
    if profile is not None:
        values["profile"] = profile
    return RuntimeSettings(**values)
