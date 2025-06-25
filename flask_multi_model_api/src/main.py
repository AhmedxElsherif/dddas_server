import os
import sys
# DON'T CHANGE THIS !!!
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from flask import Flask, send_from_directory

# Import blueprints for each model
from src.routes.sign_language import sign_language_bp
from src.routes.traffic_sign import traffic_sign_bp
from src.routes.siren_detection import siren_detection_bp

# Remove unused user model/db/routes if not needed
# from src.models.user import db
# from src.routes.user import user_bp

app = Flask(__name__, static_folder=os.path.join(os.path.dirname(__file__), 'static'))
app.config['SECRET_KEY'] = 'asdf#FGSgvasgf$5$WGT' # Keep or change as needed

# Register the blueprints for each model under the /api prefix
app.register_blueprint(sign_language_bp, url_prefix='/api/sign_language')
app.register_blueprint(traffic_sign_bp, url_prefix='/api/traffic_sign')
app.register_blueprint(siren_detection_bp, url_prefix='/api/siren_detection')

# Remove or comment out the default user blueprint if not used
# app.register_blueprint(user_bp, url_prefix='/api')

# Database configuration (commented out as not explicitly requested for models)
# app.config['SQLALCHEMY_DATABASE_URI'] = f"mysql+pymysql://{os.getenv('DB_USERNAME', 'root')}:{os.getenv('DB_PASSWORD', 'password')}@{os.getenv('DB_HOST', 'localhost')}:{os.getenv('DB_PORT', '3306')}/{os.getenv('DB_NAME', 'mydb')}"
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# db.init_app(app)
# with app.app_context():
#     db.create_all()

# Serve static files (like index.html if you add a frontend later)
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    static_folder_path = app.static_folder
    if static_folder_path is None:
            return "Static folder not configured", 404

    if path != "" and os.path.exists(os.path.join(static_folder_path, path)):
        return send_from_directory(static_folder_path, path)
    else:
        index_path = os.path.join(static_folder_path, 'index.html')
        if os.path.exists(index_path):
            return send_from_directory(static_folder_path, 'index.html')
        else:
            # If no index.html, maybe return a simple API status or message
            return "API Server Running. Access model endpoints under /api/", 200
            # return "index.html not found", 404


if __name__ == '__main__':
    # Important: Use host='0.0.0.0' to make the server accessible externally
    # Use debug=False for production or when sharing access
    app.run(host='0.0.0.0', port=5000, debug=True) # debug=True is helpful for development

