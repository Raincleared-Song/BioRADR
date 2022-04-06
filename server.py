from collection import init_all
from router import app, initialize


if __name__ == "__main__":
    print('Initializing search engine ......')
    init_all()
    print('Initialized!')
    initialize()

    app.run(host="0.0.0.0", port=17201, debug=True, threaded=True, use_reloader=False)
