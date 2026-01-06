# run.py
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["train", "test", "api"])
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config yaml")
    args = parser.parse_args()

    if args.mode == "train":
        from training.pretrain_tcn import pretrain_tcn
        pretrain_tcn(config_path=args.config)

    elif args.mode == "test":
        from training.evaluate_test import evaluate_test
        evaluate_test(config_path=args.config)

    elif args.mode == "api":
        from api.server import run_api
        run_api(config_path=args.config)


if __name__ == "__main__":
    main()
