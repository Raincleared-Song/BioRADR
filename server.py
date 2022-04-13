import warnings
from collection import init_all
from router import app, initialize


if __name__ == "__main__":
    print('Initializing search engine ......')
    init_all()
    print('Initialized!')
    initialize()

    warnings.filterwarnings('default', message=r'.*scispacy/candidate_generation.py.*')

    app.run(host="0.0.0.0", port=17201, debug=True, threaded=True, use_reloader=False)
