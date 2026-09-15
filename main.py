import argparse
from pathlib import Path

from handtracking_runtime import run
from handtracking_settings import PROFILE_SCALES, load_settings


def main(argv=None):
    parser = argparse.ArgumentParser(description="Controllo gestuale locale")
    parser.add_argument("--config", type=Path, help="Impostazioni JSON")
    parser.add_argument("--profile", choices=PROFILE_SCALES)
    trace = parser.add_mutually_exclusive_group()
    trace.add_argument("--record", type=Path, help="Registra metadati locali, senza video")
    trace.add_argument("--replay", type=Path, help="Verifica una traccia senza webcam o input OS")
    parser.add_argument("--record-max-frames", type=int, default=1800)
    args = parser.parse_args(argv)
    if not 1 <= args.record_max_frames <= 18000:
        parser.error("--record-max-frames deve essere tra 1 e 18000")
    if args.replay is not None:
        if args.config is not None or args.profile is not None:
            parser.error("Replay usa le impostazioni della traccia, non --config/--profile")
        from handtracking_trace import replay_trace
        try:
            result = replay_trace(args.replay)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        if not result["verified"]:
            parser.error("Replay non verificato: " + "; ".join(result["mismatches"][:3]))
        print(f"Replay: {result['frames']} frame | fonte {result['source']} | verificato {result['verified']}")
        return result
    config = args.config
    if config is None:
        local_config = Path(__file__).with_name("handtracking.json")
        config = local_config if local_config.exists() else None
    try:
        settings = load_settings(config, args.profile)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return run(settings=settings, record_path=args.record,
               record_max_frames=args.record_max_frames)


if __name__ == "__main__":
    main()
