from utils.utils import load_setting, init


def main():
    # Load Setting (setting.json)
    setting = load_setting()

    # Initialization setting and logger
    setting, logger = init(setting)

    return


if __name__ == "__main__":
    main()
