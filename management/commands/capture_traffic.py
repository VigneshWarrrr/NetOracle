from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from feature_engine.services import ingest_live_events
from ingestion.live_capture import capture_forever


class Command(BaseCommand):
    help = "Capture local interface traffic and feed the NetOracle world model."

    def add_arguments(self, parser):
        parser.add_argument("--iface", default=None, help="Interface name, for example Wi-Fi or Ethernet.")
        parser.add_argument("--batch-size", type=int, default=20)
        parser.add_argument("--stream-key", default="global")

    def handle(self, *args, **options):
        if options["batch_size"] <= 0:
            raise CommandError("--batch-size must be greater than zero.")

        model_path = settings.BASE_DIR / "models" / "attack_forecaster.pt"
        self.stdout.write(self.style.SUCCESS("Starting local packet capture. Press Ctrl+C to stop."))
        self.stdout.write(f"Interface: {options['iface'] or 'Scapy default'}")

        def process_batch(events):
            result = ingest_live_events(
                events,
                stream_key=options["stream_key"],
                model_path=model_path,
            )
            self.stdout.write(
                f"Captured {len(events)} packets | "
                f"risk={result['forecasted_risk']:.2f} | "
                f"attack={result['attack_type']}"
            )

        try:
            capture_forever(options["iface"], options["batch_size"], process_batch)
        except PermissionError as error:
            raise CommandError(
                "Packet capture permission denied. Install Npcap and run the terminal as Administrator."
            ) from error
        except OSError as error:
            raise CommandError(
                f"Could not open interface {options['iface'] or '(default)'}: {error}"
            ) from error
