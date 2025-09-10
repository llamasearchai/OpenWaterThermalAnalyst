import json
import pytest
from unittest.mock import patch, MagicMock

from open_water_thermal_analyst.alerts import send_alert


class TestAlertSystem:
    """Test alert system functionality."""

    def test_send_alert_with_webhook(self):
        """Test sending alert with webhook URL."""
        with patch('requests.post') as mock_post:
            mock_response = MagicMock()
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response

            result = send_alert(
                level="WARNING",
                message="Test warning message",
                context={"temperature": 30.5, "location": "test_site"},
                webhook_url="https://test-webhook.com/alerts"
            )

            # Verify the function doesn't return anything (implicitly successful)
            assert result is None

            # Verify requests.post was called correctly
            mock_post.assert_called_once_with(
                "https://test-webhook.com/alerts",
                json={
                    "level": "WARNING",
                    "message": "Test warning message",
                    "context": {"temperature": 30.5, "location": "test_site"}
                },
                timeout=10
            )

    def test_send_alert_without_webhook(self):
        """Test sending alert without webhook (should print to stdout)."""
        with patch('builtins.print') as mock_print:
            result = send_alert(
                level="INFO",
                message="Test info message",
                context={"sensor_id": "sensor_001"}
            )

            assert result is None

            # Verify print was called with JSON
            mock_print.assert_called_once()
            printed_arg = mock_print.call_args[0][0]
            printed_data = json.loads(printed_arg)

            assert printed_data["alert"]["level"] == "INFO"
            assert printed_data["alert"]["message"] == "Test info message"
            assert printed_data["alert"]["context"]["sensor_id"] == "sensor_001"

    def test_send_alert_webhook_failure(self):
        """Test alert sending when webhook fails."""
        with patch('requests.post') as mock_post:
            mock_post.side_effect = Exception("Connection failed")

            with patch('builtins.print') as mock_print:
                result = send_alert(
                    level="CRITICAL",
                    message="Critical system failure",
                    webhook_url="https://failing-webhook.com/alerts"
                )

                assert result is None

                # Verify fallback print was called
                mock_print.assert_called_once()
                printed_arg = mock_print.call_args[0][0]
                printed_data = json.loads(printed_arg)

                assert printed_data["alert"]["level"] == "CRITICAL"
                assert "error" in printed_data
                assert "Connection failed" in printed_data["error"]

    def test_send_alert_with_env_webhook(self):
        """Test sending alert using webhook from environment variable."""
        with patch('requests.post') as mock_post:
            with patch('os.environ.get') as mock_env_get:
                mock_env_get.return_value = "https://env-webhook.com/alerts"
                mock_response = MagicMock()
                mock_response.raise_for_status.return_value = None
                mock_post.return_value = mock_response

                result = send_alert(
                    level="WARNING",
                    message="Environment webhook test"
                )

                assert result is None
                mock_post.assert_called_once_with(
                    "https://env-webhook.com/alerts",
                    json={
                        "level": "WARNING",
                        "message": "Environment webhook test",
                        "context": {}
                    },
                    timeout=10
                )

    def test_send_alert_timeout(self):
        """Test alert sending with timeout."""
        with patch('requests.post') as mock_post:
            mock_post.side_effect = Exception("Timeout")

            with patch('builtins.print') as mock_print:
                result = send_alert(
                    level="CRITICAL",
                    message="Timeout test",
                    webhook_url="https://timeout-webhook.com/alerts"
                )

                assert result is None

                # Verify error handling
                printed_arg = mock_print.call_args[0][0]
                printed_data = json.loads(printed_arg)
                assert "error" in printed_data
                assert "Timeout" in printed_data["error"]

    def test_send_alert_different_levels(self):
        """Test alert sending with different severity levels."""
        test_cases = [
            ("INFO", "Informational message"),
            ("WARNING", "Warning condition detected"),
            ("CRITICAL", "Critical system alert"),
            ("ERROR", "Error occurred"),
        ]

        for level, message in test_cases:
            with patch('builtins.print') as mock_print:
                result = send_alert(level=level, message=message)
                assert result is None

                printed_arg = mock_print.call_args[0][0]
                printed_data = json.loads(printed_arg)
                assert printed_data["alert"]["level"] == level
                assert printed_data["alert"]["message"] == message

    def test_send_alert_empty_context(self):
        """Test alert sending with empty context."""
        with patch('builtins.print') as mock_print:
            result = send_alert(
                level="INFO",
                message="No context message"
            )

            assert result is None

            printed_arg = mock_print.call_args[0][0]
            printed_data = json.loads(printed_arg)
            assert printed_data["alert"]["context"] == {}

    def test_send_alert_large_context(self):
        """Test alert sending with large context data."""
        large_context = {
            "sensor_data": {"temp": 25.5, "humidity": 65.2, "pressure": 1013.25},
            "location": {"lat": 40.7128, "lon": -74.0060, "elevation": 10.5},
            "metadata": {"sensor_id": "THERMAL_001", "timestamp": "2024-01-01T12:00:00Z"},
            "analysis": {
                "mean_temp": 24.8,
                "std_temp": 2.1,
                "trend": "increasing",
                "anomalies": [28.5, 29.2, 27.8]
            }
        }

        with patch('builtins.print') as mock_print:
            result = send_alert(
                level="WARNING",
                message="Complex analysis alert",
                context=large_context
            )

            assert result is None

            printed_arg = mock_print.call_args[0][0]
            printed_data = json.loads(printed_arg)
            assert printed_data["alert"]["context"] == large_context

    def test_send_alert_special_characters(self):
        """Test alert sending with special characters in message and context."""
        special_message = "Temperature exceeded threshold: 32.5°C → action required!"
        special_context = {
            "notes": "Sensor malfunction detected @ location #001",
            "actions": ["notify_operator", "log_incident", "escalate_if_needed"],
            "symbols": {"delta": "Δ", "degree": "°", "arrow": "→"}
        }

        with patch('builtins.print') as mock_print:
            result = send_alert(
                level="CRITICAL",
                message=special_message,
                context=special_context
            )

            assert result is None

            printed_arg = mock_print.call_args[0][0]
            printed_data = json.loads(printed_arg)
            assert printed_data["alert"]["message"] == special_message
            assert printed_data["alert"]["context"] == special_context
