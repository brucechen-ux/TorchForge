from __future__ import annotations

from _common import parse_train_args, train_model, write_json


def main() -> None:
    train_config, components, variant = parse_train_args()
    result = train_model(
        train_config=train_config,
        components=components,
        variant=variant,
    )
    write_json(result, train_config.output)
    print(f"wrote {train_config.output}")


if __name__ == "__main__":
    main()
