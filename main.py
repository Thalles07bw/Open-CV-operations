import numpy as np
import cv2

#funcao para calcular o centro do contorno
def center(x, y, w, h):
    x1 = int(w/2)
    y1 = int(h/2)
    cx = x + x1
    cy = y + y1
    return cx,cy

#Opcoes das linhas
posL = 250
offSet = 50

xy1 = (120, posL)
xy2 = (720, posL)

######
#variaveis
detects = []
#contagem
total = 0
up = 0
down = 0

cap = cv2.VideoCapture('1.mp4')
fgbg = cv2.createBackgroundSubtractorMOG2()

while 1:
    ret, frame = cap.read()    

    #converter imagem para cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Gerar a mascara
    fgmask = fgbg.apply(gray)

    #remover sombras da mascara
    retval, th = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)

    #remover noise da mascara
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    opening = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations= 2)

    #diltar os objetos que restaram 
    dilation = cv2.dilate(opening, kernel, iterations= 8)

    #fechar o reconhecimento de um objeto
    closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel, iterations= 8)

    #retirar contorno
    contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #gerar linhas
    cv2.line(frame, xy1, xy2,(255,0,0),3)
    cv2.line(frame,(xy1[0], posL - offSet), (xy2[0], posL - offSet), (255,255,0),2)
    cv2.line(frame,(xy1[0], posL + offSet), (xy2[0], posL + offSet), (255,255,0),2)

    i = 0 #id da contagem
    for cnt in contours:
        #calcular area
        (x, y, w, h) = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)

        #limite do objeto
        if int(area) > 3000:
            #calcular centro
            centro = center(x, y, w, h)
            #gerar contorno e circulo
            cv2.circle(frame, centro, 4, (0,0,255), -1)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0),2)
            #gerar codigo do objeto
            cv2.putText(frame, str(i), (x+5, y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)    

            #incrementar id 
            if len(detects) <= i:
                detects.append([])
            if centro[1]> posL - offSet and centro[1] < posL + offSet:
                detects[i].append(centro)
            else:
                detects[i].clear()
                
            i += 1
    
    #limpar a lista
    if len(contours) == 0:
        detects.clear()
    else:
        for detect in detects:
            for (c,l) in enumerate(detect):
                #verificar se o objeto subiu
                if detect[c-1][1] < posL and l[1] > posL :
                    detect.clear()
                    up+=1
                    total+=1
                    cv2.line(frame,xy1,xy2,(0,255,0),5)
                    continue
                #verifica se o objeto desceu
                if detect[c-1][1] > posL and l[1] < posL:
                    detect.clear()
                    down+=1
                    total+=1
                    cv2.line(frame,xy1,xy2,(0,0,255),5)
                    continue
                #desenha a linha que segue o objeto
                if c > 0:
                    cv2.line(frame,detect[c-1],l,(0,0,255),1)

    #print(detects)
    
    #####
    #exibicoes
    cv2.putText(frame, "Total: " + str(total), (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255),2)
    cv2.putText(frame, "Subindo: " + str(up), (10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),2)
    cv2.putText(frame, "Descendo: " + str(down), (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),2)
    #video
    cv2.imshow("frame", frame)
    cv2.imshow("grayFrame", gray)
    

    if cv2.waitKey(90) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()