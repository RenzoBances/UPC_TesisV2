# 1.1. PYTHON LIBRARIES
#######################
import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import time
import base64
from random import randrange
import pandas as pd
import pickle
from math import acos, degrees


#1.2. OWN LIBRARIES
###################
import Libraries.Exercises.UpcSystemCost as UpcSystemCost


# 2. FUNCTIONS
##############
@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def load_exercise_metadata(id_exercise):
    df = pd.read_csv('02. trainers/exercises_metadata.csv', sep = ';')
    st.session_state.short_name = df.loc[df['id_exercise']==id_exercise, 'short_name'].values[0]
    st.session_state.vista = df.loc[df['id_exercise']==id_exercise, 'vista'].values[0]
    st.session_state.detail = df.loc[df['id_exercise']==id_exercise, 'detail'].values[0]
    st.session_state.n_poses = df.loc[df['id_exercise']==id_exercise, 'n_poses'].values[0]

    st.session_state.n_sets_default = df.loc[df['id_exercise']==id_exercise, 'n_sets_default'].values[0]
    st.session_state.n_reps_default = df.loc[df['id_exercise']==id_exercise, 'n_reps_default'].values[0]
    st.session_state.n_rest_time_default = df.loc[df['id_exercise']==id_exercise, 'n_rest_time_default'].values[0]

def get_exercise_gif(id_exercise):
    gif_file = "02. trainers/" + id_exercise + "/images/" + id_exercise + ".gif"
    return gif_file

def font_size_px(markdown_text):
    return "<span style='font-size:26px'>" + markdown_text + "</span>"

def load_home():
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("**POSE_LANDMARKS**<br>Una lista de puntos de referencia de la pose. Cada punto de referencia consta de lo siguiente:<br><ul><li><b>X & Y:</b> coordenadas de referencia normalizadas a [0.0, 1.0] por el ancho y la altura de la imagen, respectivamente.</li><li><b>Z:</b> Representa la profundidad del punto de referencia con la profundidad en el punto medio de las caderas como origen, y cuanto menor sea el valor, más cerca estará el punto de referencia de la cámara. La magnitud de z usa aproximadamente la misma escala que x.</li><li><b>Visibilidad:</b> un valor en [0.0, 1.0] que indica la probabilidad de que el punto de referencia sea visible (presente y no ocluido) en la imagen.</li></ul><br>",
        unsafe_allow_html=True)
    st.markdown("**MODELO DE PUNTOS DE REFERENCIA DE POSE (BlazePose GHUM 3D)**<br>El modelo de puntos de referencia en MediaPipe Pose predice la ubicación de 33 puntos de referencia de pose (consulte la figura a continuación).<br>",
        unsafe_allow_html=True)
    st.image("01. webapp_img/pose_landmarks_model.png", width=600)

def print_sidebar_main(id_exercise):
        load_exercise_metadata(id_exercise)

        #SIDEBAR START
        st.sidebar.markdown('---')
        st.sidebar.markdown(f'''**{st.session_state.short_name}**''', unsafe_allow_html=True)
        st.sidebar.image(get_exercise_gif(id_exercise))  
        vista_gif = '01. webapp_img/vista_' + st.session_state.vista + '.gif'
        with st.sidebar.expander("💡 Info"):
            st.info(st.session_state.detail)
        st.sidebar.markdown("<hr/>", unsafe_allow_html=True)
        st.session_state.n_sets = st.sidebar.number_input("Sets", min_value=1, max_value=10, value=st.session_state.n_sets_default)
        st.session_state.n_reps = st.sidebar.number_input("Reps", min_value=1, max_value=10, value=st.session_state.n_reps_default)
        st.session_state.seconds_rest_time = st.sidebar.number_input("Rest Time (seconds)", min_value=1, max_value=30, value=st.session_state.n_rest_time_default)
        position_image, position_text = st.sidebar.columns(2)
        with position_image:
            st.image(vista_gif, width=100)
        with position_text:
            st.markdown("<hr/>", unsafe_allow_html=True)
            st.markdown("**Vista:** " + st.session_state.vista, unsafe_allow_html=True)
            st.markdown("**N° poses:** " + str(st.session_state.n_poses), unsafe_allow_html=True)
        exercise_to_do[app_mode] = {"reps":st.session_state.n_reps,"sets":st.session_state.n_sets,"secs":st.session_state.seconds_rest_time}
        st.sidebar.markdown("<hr/>", unsafe_allow_html=True)
        placeholder_title.title('STARTER TRAINING - '+ st.session_state.short_name)
        st.markdown('---')
        
def get_trainer_coords(id_exercise, id_trainer):
        df = pd.read_csv("02. trainers/" + id_exercise + "/costs/" + id_exercise + "_puntos_trainer"+str(id_trainer)+".csv")
        del df['pose']
        del df['right_torso_angles']
        del df['right_arm_angles']
        del df['right_leg_angles']
        return df

def get_cost_pose_trainer(id_exercise, n_pose):
    if n_pose >= 1:
        df = pd.read_csv("02. trainers/" + id_exercise + "/costs/costos_" + id_exercise + "_promedio.csv")

        cost_align = df.loc[df['Pose'] == n_pose,"Costo_alineamiento"].reset_index(drop = True)[0]
        ds = df.loc[df['Pose'] == n_pose,"Desviacion_estandar"].reset_index(drop = True)[0]
        
        pose_trainer_cost_min = round(cost_align - ds, 2)
        pose_trainer_cost_max = round(cost_align + ds, 2)
    else:
        pose_trainer_cost_min = 0
        pose_trainer_cost_max = 0
    return pose_trainer_cost_min, pose_trainer_cost_max

def get_trainers_angles(id_exercise):
    df = pd.read_csv("02. trainers/" + id_exercise + "/costs/angulos_" + id_exercise + "_promedio.csv")
    return df

def LoadModel():
    model_weights = './04. model_weights/squats/weights_body_language.pkl'
    with open(model_weights, "rb") as f:
        model = pickle.load(f)
    return model

def calculate_angleacos(a,b,c):
    angle = degrees(acos((a**2 + c**2 - b**2) / (2 * a * c)))
    angle = int(angle)
    return angle

def get_angle(df, index, part):
    angle_in=df['Angulo'][(df.pose==index+1)&(df.Parte==part)]
    angle_in=angle_in.iloc[0]
    return angle_in

def get_desv_angle(df, index, part):
    desv_in=df['Desviacion_estandar'][(df.pose==index+1)&(df.Parte==part)]
    desv_in=desv_in.iloc[0]
    return desv_in

def next_pose(actual_pose):
    if actual_pose < st.session_state.n_poses:
        next_pose_n = actual_pose + 1
    else:
        next_pose_n = 1   
    return next_pose_n

def update_dashboard():
    
        if st.session_state.count_pose_g < st.session_state.total_poses:
            st.session_state.count_pose_g += 1
            placeholder_status.markdown(font_size_px("🏎️ TRAINING..."), unsafe_allow_html=True)
            placeholder_pose_global.metric("POSE GLOBAL", str(st.session_state.count_pose_g) + " / " + str(st.session_state.total_poses), "+1 pose")

            if st.session_state.count_pose < st.session_state.n_poses:
                st.session_state.count_pose += 1
                placeholder_pose.metric("POSE", str(st.session_state.count_pose) + " / "+ str(st.session_state.n_poses), "+1 pose")
                placeholder_next.image("./02. trainers/" + id_exercise + "/images/" + id_exercise + str(next_pose(st.session_state.count_pose)) + ".png")
                placeholder_trainer.image("./02. trainers/" + id_exercise + "/images/" + id_exercise + str(st.session_state.count_pose) + ".png")
        else:
            placeholder_status.markdown(font_size_px("🥇 FINISH !!!"), unsafe_allow_html=True)
            placeholder_trainer.image("./02. trainers/" + id_exercise + "/images/" + id_exercise + "1.png")
            placeholder_pose_global.metric("POSE GLOBAL", str(st.session_state.count_pose_g) + " / " + str(st.session_state.total_poses), "COMPLETED", delta_color="inverse")
            placeholder_pose.metric("POSE", str(st.session_state.n_poses) + " / "+ str(st.session_state.n_poses), "COMPLETED", delta_color="inverse")
            placeholder_rep.metric("REPETITION", str(st.session_state.count_rep) + " / "+ str(st.session_state.n_reps), "COMPLETED", delta_color="inverse")
            placeholder_set.metric("SET", str(st.session_state.count_set) + " / "+ str(st.session_state.n_sets), "COMPLETED", delta_color="inverse" )

# 3. HTML CODE
#############
st.set_page_config(
    page_title="STARTER TRAINING - UPC",
    page_icon ="01. webapp_img/upc_logo.png",
)

img_upc = get_base64_of_bin_file('01. webapp_img/upc_logo_50x50.png')
fontProgress = get_base64_of_bin_file('01. webapp_fonts/ProgressPersonalUse-EaJdz.ttf')

st.markdown(
    f"""
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {{
        width: 336px;        
    }}
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {{
        width: 336px;
        margin-left: -336px;
        background-color: #00F;
    }}
    [data-testid="stVerticalBlock"] {{
        flex: 0;
        gap: 0;
    }}
    #starter-training{{
        padding: 0;
    }}
    #div-upc{{
        #border: 1px solid #DDDDDD;
        background-image: url("data:image/png;base64,{img_upc}");
        position: fixed !important;
        right: 14px;
        top: 60px;
        width: 50px;
        height: 50px;
        background-size: 50px 50px;
    }}
    @font-face {{
        font-family: ProgressFont;
        src: url("data:image/png;base64,{fontProgress}");
    }}
    .css-10trblm{{
        color: #FFF;
        font-size: 40px;
        font-family: ProgressFont;    
    }}
    .main {{
        background: linear-gradient(135deg,#a8e73d,#09e7db,#092de7);
        background-size: 180% 180%;
        animation: gradient-animation 3s ease infinite;
        }}

        @keyframes gradient-animation {{
            0% {{ background-position: 0% 50%; }}
            50% {{ background-position: 100% 50%; }}
            100% {{ background-position: 0% 50%; }}
        }}
    .block-container{{
        max-width: 100%;
    }}
    .css-17qbjix {{
        font-size: 16px;
    }}
    .css-12oz5g7 {{
        padding-top: 3rem;
    }}
    .stButton{{
        text-align: center !important;
    }}
    </style>
    """ ,
    unsafe_allow_html=True,
)


# 4. PYTHON CODE
#############
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

if 'camera' not in st.session_state:
    st.session_state['camera'] = 0

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

placeholder_title = st.empty()
placeholder_title.title('STARTER TRAINING')

st.markdown("<div id='div-upc'></span>", unsafe_allow_html=True)

app_mode = st.sidebar.selectbox('Choose your training:',
    ['HOME','Squats', 'Push Up', 'Curl Up', 'Front Plank', 'Forward Lunge', 'Bird Dog']
)

id_trainer = randrange(3) + 1

exercise_to_do = {}

if app_mode =='HOME':
    load_home()

else:
    if app_mode =='Squats':
        id_exercise = 'squats'

    elif app_mode =='Push Up':
        id_exercise = 'push_up'

    elif app_mode =='Curl Up':
        id_exercise = 'curl_up'

    elif app_mode =='Front Plank':
        id_exercise = 'front_plank'

    elif app_mode =='Forward Lunge':
        id_exercise = 'forward_lunge'

    elif app_mode =='Bird Dog':
        id_exercise = 'bird_dog'

    else:
        id_exercise = None

    print_sidebar_main(id_exercise)

    #MAIN-SCREEN START
    st.session_state.count_pose_g = 0
    st.session_state.count_pose   = 0
    st.session_state.count_rep    = 0
    st.session_state.count_set    = 0

    # total_poses = Sets x Reps x N° Poses
    st.session_state.total_poses = st.session_state.n_sets * st.session_state.n_reps * st.session_state.n_poses
    exercise_control, exercise_next, exercise_number_set, exercise_number_rep, exercise_number_pose, exercise_number_pose_global, exercise_status  = st.columns(7)
        
    with exercise_control:
        placeholder_button_status = st.empty()
        placeholder_button_status.info('PRESS START BUTTON', icon="📹")
        st.markdown("<br>", unsafe_allow_html=True)
        webcam = st.button("START / STOP")
    
    with exercise_next:
        st.text("NEXT POSE")
        placeholder_next = st.empty()
        placeholder_next.image("./02. trainers/" + id_exercise + "/images/" + id_exercise + str(next_pose(st.session_state.count_pose)) + ".png")

    with exercise_number_set:
        placeholder_set = st.empty()
        placeholder_set.metric("SET", "0 / "+ str(st.session_state.n_sets), "+1 set")

    with exercise_number_rep:
        placeholder_rep = st.empty()
        placeholder_rep.metric("REPETITION", "0 / "+ str(st.session_state.n_reps), "+1 repetition")

    with exercise_number_pose:
        placeholder_pose = st.empty()
        placeholder_pose.metric("POSE", "0 / "+ str(st.session_state.n_poses), "+1 pose")
    
    with exercise_number_pose_global:
        placeholder_pose_global = st.empty()
        placeholder_pose_global.metric("POSE GLOBAL", "0 / " + str(st.session_state.total_poses), "+1 pose")

    with exercise_status:
        placeholder_status = st.empty()
        st.markdown("<br>", unsafe_allow_html=True)
        placeholder_status.markdown(font_size_px("⛽ READY?"), unsafe_allow_html=True)

    st.markdown('---')

    trainer, user = st.columns(2)

    with trainer:        
        st.markdown("**TRAINER**", unsafe_allow_html=True)
        placeholder_trainer = st.empty()
        placeholder_trainer.image("./01. webapp_img/trainer.png")

    with user:
        st.markdown("**USER**", unsafe_allow_html=True)
        stframe = st.empty()
        #stframe.image("./01. webapp_img/user.png")

        #Cris-DM pasar a función - inicio
        df_trainer_coords = get_trainer_coords(id_exercise, id_trainer)
        df_trainers_angles = get_trainers_angles(id_exercise)
        #Cris-DM pasar a función - fin
    
    st.markdown('---')
    placeholder_results_1 = st.empty()
    placeholder_results_2 = st.empty()
    
    with exercise_control:
        if(webcam):
            video_capture = cv2.VideoCapture(0,cv2.CAP_DSHOW)

            # Cámara apagada
            if st.session_state['camera'] % 2 != 0:
                placeholder_button_status.warning('CAMERA OFF ⚫', icon="📹")
                st.session_state['camera'] += 1
                video_capture.release()
                cv2.destroyAllWindows()
                stframe.image("./01. webapp_img/user.png")

            # Cámara encendida
            else: 
                placeholder_button_status.success('CAMERA ON  🔴', icon="📹")
                st.session_state['camera'] += 1

                st.session_state.count_set = 0
                st.session_state.count_rep = 0
                
                
                N = 5
                placeholder_trainer.image("./01. webapp_img/warm_up.gif")
                stframe.image("./01. webapp_img/warm_up.gif")
                for secs in range(N,0,-1):
                    ss = secs%60
                    placeholder_status.markdown(font_size_px(f"🏁 START IN {ss:02d}"), unsafe_allow_html=True)
                    time.sleep(1)

                placeholder_trainer.image("./02. trainers/" + id_exercise + "/images/" + id_exercise + "1.png")

                with user:
                    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
                    up = False
                    down = False
                    start = 0
                    
                    while st.session_state.count_set < st.session_state.n_sets:
                        stage = ""
                        
                        st.session_state.count_rep = 0
                        # Setup mediapipe instance
                        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
                            cap.isOpened()
                            # while st.session_state.count_pose_g <= st.session_state.total_poses:
                            while st.session_state.count_rep < st.session_state.n_reps:
                                ret, frame = cap.read()
                                if ret == False:
                                    break
                                # frame = cv2.flip(frame,1)
                                height, width, _ = frame.shape
                                # Recolor image to RGB
                                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                image.flags.writeable = False
                            
                                # Make detection
                                results = pose.process(image)

                                # Recolor back to BGR
                                image.flags.writeable = True
                                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                                
                                # Extract landmarks
                                # try:
                                if results.pose_landmarks is None:
                                    cv2.putText(image, 
                                    "No se han detectado ninguno de los 33 puntos corporales",
                                    (100,250),
                                    cv2.FONT_HERSHEY_DUPLEX,
                                    0.5,
                                    (0, 0, 255),
                                    1, 
                                    cv2.LINE_AA)
                                    stframe.image(image,channels = 'BGR',use_column_width=True)   
                                        
                                else:
                                    ############################################################
                                    ##         🏃‍♀️ SISTEMA PREDICCIÓN EJERCICIO (INICIO)       ##
                                    ############################################################

                                    landmarks = results.pose_landmarks.landmark
                                    pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in landmarks]).flatten())

                                    # Concate rows
                                    row = pose_row

                                    # Make Detections
                                    X = pd.DataFrame([row])

                                    # Load Model Clasification
                                    body_language_class = LoadModel().predict(X)[0]
                                    body_language_prob = LoadModel().predict_proba(X)[0]
                                    body_language_prob1 = body_language_prob*100
                                    body_language_prob1=round(body_language_prob1[np.argmax(body_language_prob1)],2)

                                    ############################################################
                                    ##         🏃‍♀️ SISTEMA PREDICCIÓN EJERCICIO (FIN)          ##
                                    ############################################################

                                    right_arm_x1 = int(landmarks[12].x * width) #right_arm
                                    right_arm_x2 = int(landmarks[14].x * width)
                                    right_arm_x3 = int(landmarks[16].x * width)
                                    right_arm_y1 = int(landmarks[12].y * height)
                                    right_arm_y2 = int(landmarks[14].y * height)
                                    right_arm_y3 = int(landmarks[16].y * height)  

                                    right_arm_p1 = np.array([right_arm_x1, right_arm_y1])
                                    right_arm_p2 = np.array([right_arm_x2, right_arm_y2])
                                    right_arm_p3 = np.array([right_arm_x3, right_arm_y3])

                                    right_arm_l1 = np.linalg.norm(right_arm_p2 - right_arm_p3)
                                    right_arm_l2 = np.linalg.norm(right_arm_p1 - right_arm_p3)
                                    right_arm_l3 = np.linalg.norm(right_arm_p1 - right_arm_p2)

                                    # Calculate angle
                                    right_arm_angle = calculate_angleacos(right_arm_l1, right_arm_l2, right_arm_l3)
                                    print(f'right_arm_angle: {right_arm_angle}')

                                    right_torso_x1 = int(landmarks[12].x * width) #right_torso
                                    right_torso_x2 = int(landmarks[24].x * width)
                                    right_torso_x3 = int(landmarks[26].x * width) 
                                    right_torso_y1 = int(landmarks[12].y * height)
                                    right_torso_y2 = int(landmarks[24].y * height)
                                    right_torso_y3 = int(landmarks[26].y * height) 

                                    right_torso_p1 = np.array([right_torso_x1, right_torso_y1])
                                    right_torso_p2 = np.array([right_torso_x2, right_torso_y2])
                                    right_torso_p3 = np.array([right_torso_x3, right_torso_y3])

                                    right_torso_l1 = np.linalg.norm(right_torso_p2 - right_torso_p3)
                                    right_torso_l2 = np.linalg.norm(right_torso_p1 - right_torso_p3)
                                    right_torso_l3 = np.linalg.norm(right_torso_p1 - right_torso_p2)

                                    # Calculate angle
                                    right_torso_angle = calculate_angleacos(right_torso_l1, right_torso_l2, right_torso_l3)
                                    print(f'right_torso_angle: {right_torso_angle}')

                                    right_leg_x1 = int(landmarks[24].x * width) #right_leg
                                    right_leg_x2 = int(landmarks[26].x * width)
                                    right_leg_x3 = int(landmarks[28].x * width) 
                                    right_leg_y1 = int(landmarks[24].y * height)
                                    right_leg_y2 = int(landmarks[26].y * height)
                                    right_leg_y3 = int(landmarks[28].y * height)

                                    right_leg_p1 = np.array([right_leg_x1, right_leg_y1])
                                    right_leg_p2 = np.array([right_leg_x2, right_leg_y2])
                                    right_leg_p3 = np.array([right_leg_x3, right_leg_y3])

                                    right_leg_l1 = np.linalg.norm(right_leg_p2 - right_leg_p3)
                                    right_leg_l2 = np.linalg.norm(right_leg_p1 - right_leg_p3)
                                    right_leg_l3 = np.linalg.norm(right_leg_p1 - right_leg_p2)

                                    # Calculate angle
                                    right_leg_angle = calculate_angleacos(right_leg_l1, right_leg_l2, right_leg_l3)
                                    print(f'right_leg_angle: {right_leg_angle}')

                                    ############################################################
                                    ##                💰 SISTEMA COSTOS (INICIO)              ##
                                    ############################################################

                                    # pose_trainer_cost_min, pose_trainer_cost_max = get_cost_pose_trainer(id_exercise, st.session_state.count_pose)
                                    # pose_user_cost = UpcSystemCost.get_cost_pose_user(df_trainer_coords, results, st.session_state.count_pose)
                                    
                                    # color_validation = (255, 0, 0) #Azul - dentro del rango
                                    # message_validation = "Correct Position"

                                    # if pose_user_cost < pose_trainer_cost_min or pose_user_cost > pose_trainer_cost_min:
                                    #     color_validation = (0, 0, 255) #Rojo - fuera del rango
                                    #     message_validation = "Wrong Position"

                                    # #1. Esquina superior izquierda: Evaluación de costos trainer vs user
                                    # cv2.rectangle(image, (700,0), (415,50), (245,117,16), -1)
                                    # cv2.putText(image, 
                                    #             "Pose: "+ str(st.session_state.count_pose),
                                    #             (435,20),
                                    #             cv2.FONT_HERSHEY_DUPLEX,
                                    #             0.5,
                                    #             (255,255,255),
                                    #             1, 
                                    #             cv2.LINE_AA)
                                    # cv2.putText(image,
                                    #             "Range: [" + str(pose_trainer_cost_min) + " - " + str(pose_trainer_cost_max) + "]", #Rango costos
                                    #             (435,40),
                                    #             cv2.FONT_HERSHEY_SIMPLEX,
                                    #             0.5,
                                    #             (255,255,255),
                                    #             1, 
                                    #             cv2.LINE_AA)

                                    # #2. Esquina superior derecha: Posición correcta/incorrecta
                                    # cv2.rectangle(image, (700,70), (415,50), (255,255,255), -1)
                                    # cv2.putText(image, 
                                    #             "User cost: " + str(pose_user_cost), #Costo resultante 
                                    #             (465,65),
                                    #             cv2.FONT_HERSHEY_SIMPLEX, 
                                    #             0.5,
                                    #             color_validation,
                                    #             1, 
                                    #             cv2.LINE_AA)

                                    ############################################################
                                    ##                💰 SISTEMA COSTOS (FIN)                 ##
                                    ############################################################


                                    
                                    ############################################################
                                    ##                📐 SISTEMA ÁNGULOS (INICIO)             ##
                                    ############################################################


                                    if body_language_class == "pushup" and body_language_prob1 > 70: 
                                        print(f'body_language_prob1: {body_language_prob1}')
                                        print(f'start: {start}')
                                        right_arm_angle_in= get_angle(df_trainers_angles, start, 'right_arm_angles')
                                        print(f'right_arm_angle_in: {right_arm_angle_in}')
                                        right_torso_angle_in=get_angle(df_trainers_angles, start, 'right_torso_angles')
                                        print(f'right_torso_angle_in: {right_torso_angle_in}')
                                        right_leg_angle_in=get_angle(df_trainers_angles, start, 'right_leg_angles')
                                        print(f'right_leg_angle_in: {right_leg_angle_in}')
                                        desv_right_arm_angle_in=10#get_desv_angle(df_trainers_angles, start, 'right_arm_angles')
                                        print(f'desv_right_arm_angle: {desv_right_arm_angle_in}')
                                        desv_right_torso_angle_in=10#get_desv_angle(df_trainers_angles, start, 'right_torso_angles')
                                        print(f'desv_right_torso_angle: {desv_right_torso_angle_in}')
                                        desv_right_leg_angle_in=10#get_desv_angle(df_trainers_angles, start, 'right_leg_angles')
                                        print(f'desv_right_leg_angle: {desv_right_leg_angle_in}')

                                        #SUMAR Y RESTAR UN RANGO DE 30 PARA EL ANGULO DE CADA POSE PARA UTILIZARLO COMO RANGO 
                                        if  up == False and right_arm_angle in range(int(right_arm_angle_in-desv_right_arm_angle_in), 
                                        int(right_arm_angle_in + desv_right_arm_angle_in + 1)) and right_torso_angle in range(int(right_torso_angle_in - desv_right_torso_angle_in), int(right_torso_angle_in + desv_right_torso_angle_in + 1)) and right_leg_angle in range(int(right_leg_angle_in - desv_right_leg_angle_in),int(right_leg_angle_in + desv_right_leg_angle_in + 1)):
                                            up = True
                                            stage = "up"
                                            ############################################
                                            update_dashboard()
                                            ############################################ 
                                            print(f'Paso Primera Pose')
                                        elif up == True and down == False and right_arm_angle in range(int(right_arm_angle_in - desv_right_arm_angle_in) , int(right_arm_angle_in + desv_right_arm_angle_in + 1)) and right_torso_angle in range(int(right_torso_angle_in - desv_right_torso_angle_in), int(right_torso_angle_in + desv_right_torso_angle_in + 1)) and right_leg_angle in range(int(right_leg_angle_in - desv_right_leg_angle_in),int(right_leg_angle_in + desv_right_leg_angle_in + 1)):
                                            down = True
                                            stage = "down"
                                            ############################################
                                            update_dashboard()
                                            ############################################ 
                                            print(f'Paso Segunda Pose')
                                        elif up == True and down == True and right_arm_angle in range(int(right_arm_angle_in - desv_right_arm_angle_in) , int(right_arm_angle_in + desv_right_arm_angle_in + 1)) and right_torso_angle in range(int(right_torso_angle_in - desv_right_torso_angle_in), int(right_torso_angle_in + desv_right_torso_angle_in + 1)) and right_leg_angle in range(int(right_leg_angle_in - desv_right_leg_angle_in),int(right_leg_angle_in + desv_right_leg_angle_in + 1)):                      
                                            print(f'Paso Tercera Pose')
                                            st.session_state.count_rep += 1
                                            up = False
                                            down = False
                                            stage = "up"
                                            start +=1
                                            ############################################
                                            update_dashboard()
                                            ######################################s######
                                            placeholder_rep.metric("REPETITION", str(st.session_state.count_rep) + " / "+ str(st.session_state.n_reps), "+1 rep")
                                            st.session_state.count_pose = 0 

                                    else:
                            
                                        stage = ""
                                        start = 0
                                        up = False
                                        print(f'Salio')
                                    # Setup status box
                                    cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
                                    
                                    # Set data
                                    cv2.putText(image, 'SET', (15,20), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2, cv2.LINE_AA)
                                    cv2.putText(image, str(st.session_state.count_set), 
                                                (10,60), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2, cv2.LINE_AA)

                                    # Rep data
                                    cv2.putText(image, 'REPS', (65,20), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2, cv2.LINE_AA)
                                    cv2.putText(image, str(st.session_state.count_rep), 
                                                (60,60), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2, cv2.LINE_AA)
                                    
                                    # Stage data
                                    cv2.putText(image, 'STAGE', (115,20), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2, cv2.LINE_AA)
                                    cv2.putText(image, stage, 
                                                (110,60), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2, cv2.LINE_AA)

                                    # Class data
                                    cv2.putText(image, 'CLASS', (15,427), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2, cv2.LINE_AA)
                                    cv2.putText(image, str(body_language_class), 
                                                (10,467), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2, cv2.LINE_AA)

                                    # Prob data
                                    cv2.putText(image, 'PROB', (125,427), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2, cv2.LINE_AA)
                                    cv2.putText(image, str(body_language_prob1), 
                                                (120,467), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2, cv2.LINE_AA)
                                    

                                    cv2.line(image, (right_arm_x1, right_arm_y1), (right_arm_x2, right_arm_y2), (242, 14, 14), 3)
                                    cv2.line(image, (right_arm_x2, right_arm_y2), (right_arm_x3, right_arm_y3), (242, 14, 14), 3)
                                    cv2.circle(image, (right_arm_x1, right_arm_y1), 6, (128, 0, 255),-1)
                                    cv2.circle(image, (right_arm_x2, right_arm_y2), 6, (128, 0, 255),-1)
                                    cv2.circle(image, (right_arm_x3, right_arm_y3), 6, (128, 0, 255),-1)
                                    cv2.putText(image, str(int(right_arm_angle)), (right_arm_x2 + 30, right_arm_y2), 1, 1.5, (128, 0, 250), 2)

                                    cv2.line(image, (right_torso_x1, right_torso_y1), (right_torso_x2, right_torso_y2), (14, 242, 59), 3)
                                    cv2.line(image, (right_torso_x2, right_torso_y2), (right_torso_x3, right_torso_y3), (14, 242, 59), 3)
                                    cv2.circle(image, (right_torso_x1, right_torso_y1), 6, (128, 0, 255),-1)
                                    cv2.circle(image, (right_torso_x2, right_torso_y2), 6, (128, 0, 255),-1)
                                    cv2.circle(image, (right_torso_x3, right_torso_y3), 6, (128, 0, 255),-1)
                                    cv2.putText(image, str(int(right_torso_angle)), (right_torso_x2 + 30, right_torso_y2), 1, 1.5, (128, 0, 250), 2)

                                    cv2.line(image, (right_leg_x1, right_leg_y1), (right_leg_x2, right_leg_y2), (14, 83, 242), 3)
                                    cv2.line(image, (right_leg_x2, right_leg_y2), (right_leg_x3, right_leg_y3), (14, 83, 242), 3)
                                    cv2.circle(image, (right_leg_x1, right_leg_y1), 6, (128, 0, 255),-1)
                                    cv2.circle(image, (right_leg_x2, right_leg_y2), 6, (128, 0, 255),-1)
                                    cv2.circle(image, (right_leg_x3, right_leg_y3), 6, (128, 0, 255),-1)
                                    cv2.putText(image, str(int(right_leg_angle)), (right_leg_x2 + 30, right_leg_y2), 1, 1.5, (128, 0, 250), 2)
                                    stframe.image(image,channels = 'BGR',use_column_width=True)
                                    
                            st.session_state.count_set += 1
                            placeholder_set.metric("SET", str(st.session_state.count_set) + " / "+ str(st.session_state.n_sets))
                            # Setup status box
                            cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
                                    
                            # Set data
                            cv2.putText(image, 'SET', (15,20), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2, cv2.LINE_AA)
                            cv2.putText(image, str(st.session_state.count_set), 
                                                (10,60), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2, cv2.LINE_AA)

                            # Rep data
                            cv2.putText(image, 'REPS', (65,20), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2, cv2.LINE_AA)
                            cv2.putText(image, str(st.session_state.count_rep), 
                                                (60,60), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2, cv2.LINE_AA)
                                    
                            # Stage data
                            cv2.putText(image, 'STAGE', (115,20), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2, cv2.LINE_AA)
                            cv2.putText(image, stage, 
                                                (110,60), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2, cv2.LINE_AA)         
                            if (st.session_state.count_set!=st.session_state.n_sets):
                                try:
                                    cv2.putText(image, 'FINISHED SET', (100,250), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 3, cv2.LINE_AA)
                                    cv2.putText(image, 'REST FOR ' + str(st.session_state.seconds_rest_time) + ' s' , (155,350), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,0,0), 3, cv2.LINE_AA)
                                    stframe.image(image,channels = 'BGR',use_column_width=True)
                                    # cv2.waitKey(1)
                                    time.sleep(int(st.session_state.seconds_rest_time))

                                except:
                                    stframe.image(image,channels = 'BGR',use_column_width=True)
                                    pass 
                    update_dashboard()                    
                    cv2.rectangle(image, (50,180), (600,300), (0,255,0), -1)
                    cv2.putText(image, 'FINISHED EXERCISE', (100,250), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3, cv2.LINE_AA)

                    stframe.image(image,channels = 'BGR',use_column_width=True)
                    time.sleep(5)          
                    cap.release()
                    cv2.destroyAllWindows()

                placeholder_button_status.warning('CAMERA OFF ⚫', icon="📹")
                st.session_state['camera'] += 1
                video_capture.release()
                cv2.destroyAllWindows()

                st.balloons()
                placeholder_results_1.markdown(font_size_px("RESULTADOS"), unsafe_allow_html=True)

                #Acá cargar dataset de resultados
                df = pd.read_csv('02. trainers/exercises_metadata.csv', sep = ';')
                placeholder_results_2.table(df)

