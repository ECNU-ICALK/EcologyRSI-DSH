"""Wait for a real asynchronous control receipt in loopback API tests."""
import time
from urllib.parse import quote

def wait_for_command(test, accepted):
    prefix = "/api" if not test.base.endswith("/api") else ""
    path = prefix + "/commands/" + quote(accepted["command_id"], safe="")
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        status, receipt = test.request(path)
        test.assertEqual(status, 200, receipt)
        if receipt["status"] == "completed":
            test.assertTrue(receipt["response"].get("remote_quiesced"), receipt)
            return 200, receipt["response"]
        time.sleep(0.01)
    raise AssertionError("native control receipt did not settle")
