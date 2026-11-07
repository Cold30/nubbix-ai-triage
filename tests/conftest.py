"""Enforce offline execution for every automated test."""

import socket

import pytest


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    """Fail before DNS resolution or socket connections can reach a provider."""
    def deny_network(*args, **kwargs):
        raise AssertionError("Network access is forbidden in automated tests")

    for name in ("getaddrinfo", "create_connection"):
        monkeypatch.setattr(socket, name, deny_network)
    for name in ("connect", "connect_ex"):
        monkeypatch.setattr(socket.socket, name, deny_network)
