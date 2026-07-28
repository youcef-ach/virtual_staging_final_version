"""Locust load testing configuration for the Virtual Staging Platform API.

Run with:
    locust -f load_test.py --host=http://localhost:8000
"""

from locust import HttpUser, task, between


class VirtualStagingUser(HttpUser):
    wait_time = between(1, 5)

    def on_start(self):
        """Log in and get token (placeholder for actual auth)."""
        # In a real scenario, we'd hit /api/auth/token here
        # and store the token in the session headers.
        pass

    @task(3)
    def view_catalog(self):
        """Simulate browsing the catalog."""
        self.client.get("/api/catalog")

    @task(2)
    def check_health(self):
        """Simulate frequent health checks."""
        self.client.get("/health")

    @task(1)
    def view_active_runs(self):
        """Simulate dashboard polling active pipeline runs."""
        self.client.get("/api/pipeline/runs/active")

    @task(1)
    def list_properties(self):
        """Simulate viewing properties."""
        self.client.get("/properties")
