import os


from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

classes = ["0","1","2","3","4","5","6","7","8","9"]
image_size = 28

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)

# uploadsフォルダが無ければ作成
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# モデル読み込み
model = load_model('./model.keras')


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        print("POST受信")  # デバッグ用

        # ファイルがあるかチェック
        if 'file' not in request.files:
            return render_template("index.html", answer="ファイルがありません")

        file = request.files['file']

        # ファイル名チェック
        if file.filename == '':
            return render_template("index.html", answer="ファイル未選択")

        # ファイル形式チェック
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            try:
                # 画像読み込み
                img = image.load_img(filepath, color_mode='grayscale',
                                     target_size=(image_size, image_size))
                img = image.img_to_array(img)

                # 正規化（超重要）
                img = img / 255.0

                # 次元追加
                data = np.array([img])

                print("shape:", data.shape)

                # 予測
                result = model.predict(data)
                predicted = np.argmax(result[0])

                pred_answer = "これは " + classes[predicted] + " です"

                return render_template("index.html", answer=pred_answer)

            except Exception as e:
                print("エラー:", e)
                return render_template("index.html", answer="画像処理でエラーが発生しました")

        else:
            return render_template("index.html", answer="対応していないファイル形式です")

    return render_template("index.html", answer="")


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    app.run(host ='0.0.0.0',port = port)