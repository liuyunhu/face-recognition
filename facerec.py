import cv2 ,sys ,numpy , os , multiprocessing
import time
from multiprocessing import Pool
size = 2

y_snflndrc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
fn_dir = 'att_faces'

#isayi = multiprocessing.cpu_count()
#print(isayi)

# resimler ve etiketlerini olusturma
(images, lables, names, id) = ([], [], {}, 0)

# üzerinde çalısılacak dosyalara erisim
for (subdirs, dirs, files) in os.walk(fn_dir):

    # herbir resim için dosyları kontrol et
    for subdir in dirs:
        names[id] = subdir
        subjectpath = os.path.join(fn_dir, subdir)

        # dosyadaki herbir resmi kontrol et
        for filename in os.listdir(subjectpath):

            # farkli dosyaları es geç
            f_name, f_extension = os.path.splitext(filename)
            if(f_extension.lower() not in
                    ['.png','.jpg','.jpeg','.gif','.pgm']):
                print(filename+", yanlis dosya tipi")
                continue
            path = subjectpath + '/' + filename
            lable = id

            # öğrenim verisine ekle
            images.append(cv2.imread(path, 0))
            lables.append(int(lable))
        id += 1
(im_width, im_height) = (112, 92)

# resimler ve etiketler için numpy dizisi oluşturma
(images, lables) = [numpy.array(lis) for lis in [images, lables]]

#OpenCV resimden model öğrenir
model = cv2.face.FisherFaceRecognizer_create()
model.train(images, lables)




def face_rec(deger):
    start_time = time.time()
    kamera = cv2.VideoCapture(deger)

    while True:

        # kamera calistigi sürece döngüde kal
        kontrol = False
        while(not kontrol):

            # kameradan okunan görüntüyü degiskene aktarma
            (kontrol, frame) = kamera.read()
            if(not kontrol):
                print("--- %s seconds ---" % (time.time() - start_time))
                sys.exit()

        frame=cv2.flip(frame,1,0) #resmi cevir

        # griye çevirme
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # hızlı tespit için yeniden boyutlandırma
        mini = cv2.resize(gray, (int(gray.shape[1] / size), int(gray.shape[0] / size)))

        faces = y_snflndrc.detectMultiScale(mini)

        for i in range(len(faces)):

            face_i = faces[i]

            (x, y, w, h) = [v * size for v in face_i]
            face = gray[y:y + h, x:x + w]
            face_resize = cv2.resize(face, (im_width, im_height))

            prediction = model.predict(face_resize)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)

            if prediction[1]<500:
                cv2.putText(frame,
                    '%s - %.0f' % (names[prediction[0]],prediction[1]),
                    (x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
            else:
                cv2.putText(frame,
                    'Bilinmiyor',
                    (x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))

        cv2.imshow("Yuz tanima", frame) #Yüz tanıma basligiyla goruntule

        deger = cv2.waitKey(10) #kontrol et
        if deger == 27: #esc
            break

if __name__ == '__main__':
    with Pool(4) as p:
        p.map(face_rec , ['faces1.mp4','faces2.mp4','faces3.mp4','faces4.mp4'])
