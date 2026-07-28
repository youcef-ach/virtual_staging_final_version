import random
from locust import HttpUser, task, between

class PipelineOperator(HttpUser):
    wait_time = between(1, 5)

    def on_start(self):
        """Fetch properties and scan versions on start to have valid IDs."""
        self.property_ids = []
        self.scan_version_ids = []

        with self.client.get("/api/properties", catch_response=True) as res:
            if res.status_code == 200:
                props = res.json()
                self.property_ids = [p["id"] for p in props]

        if self.property_ids:
            # Get scans for the first property
            prop_id = random.choice(self.property_ids)
            with self.client.get(f"/api/properties/{prop_id}/scan-versions", catch_response=True) as res:
                if res.status_code == 200:
                    scans = res.json()
                    self.scan_version_ids = [s["id"] for s in scans]

    @task(3)
    def check_runs(self):
        """Simulate polling the pipeline runs list."""
        self.client.get("/api/pipeline/runs")

    @task(2)
    def view_catalog(self):
        """Simulate fetching the catalog list."""
        self.client.get("/api/catalog/items")

    @task(1)
    def check_specific_run(self):
        """Simulate checking a specific run if any runs exist."""
        # This is a bit tricky without knowing run IDs, so we just fetch runs first, 
        # then check the detail of one of them.
        with self.client.get("/api/pipeline/runs", catch_response=True) as res:
            if res.status_code == 200:
                runs = res.json()
                if runs:
                    run_id = random.choice(runs)["id"]
                    self.client.get(f"/api/pipeline/runs/{run_id}")
