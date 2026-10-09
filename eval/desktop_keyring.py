"""Unlock the desktop keyring that holds gh's login, so the bot also works after a reboot, before anyone logs in.

GNOME Keyring keeps its collections locked until a login unlocks them. The bot unlocks the default collection (where
gh keeps its token) with the keyring passphrase, through gnome-keyring's D-Bus interface for unlocking with a
password. The passphrase goes only to the local keyring daemon. Needs python3-dbus.
"""
import dbus

SERVICE, ROOT = "org.freedesktop.secrets", "/org/freedesktop/secrets"
COLLECTION = "org.freedesktop.Secret.Collection"
INTERNAL = "org.gnome.keyring.InternalUnsupportedGuiltRiddenInterface"


def _service(bus):
    return dbus.Interface(bus.get_object(SERVICE, ROOT), "org.freedesktop.Secret.Service")


def secret(service, value):
    """A plain-session secret: the form gnome-keyring takes passwords in."""
    _, session = service.OpenSession("plain", dbus.String("", variant_level=1))
    return dbus.Struct((session, dbus.ByteArray(b""), dbus.ByteArray(value.encode()), "text/plain"),
                       signature="oayays")


def is_locked(path, bus):
    return bool(dbus.Interface(bus.get_object(SERVICE, path), dbus.PROPERTIES_IFACE).Get(COLLECTION, "Locked"))


def unlock(passphrase, path=None, bus=None):
    """Unlock the collection at `path` (default: the default collection) if it is locked. True if it was locked."""
    bus = bus or dbus.SessionBus()
    service = _service(bus)
    path = path or service.ReadAlias("default")
    if path == "/":
        raise RuntimeError("the keyring has no default collection")
    if not is_locked(path, bus):
        return False
    dbus.Interface(bus.get_object(SERVICE, ROOT), INTERNAL).UnlockWithMasterPassword(path, secret(service, passphrase))
    if is_locked(path, bus):
        raise RuntimeError("the passphrase did not unlock the keyring")
    return True
