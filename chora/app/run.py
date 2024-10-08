from pi_conf import load_config

from chora.app import create_app

cfg = load_config() # loads the .config.toml file by default
app = create_app(cfg)


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
