import os, json, base64, math, time
import numpy as np
import cv2
import mediapipe as mp
from flask import Flask, render_template, request, jsonify, session, send_file
from flask_cors import CORS
from io import BytesIO

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
    REPORTLAB_OK = True
except ImportError:
    REPORTLAB_OK = False

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'fitai_secret_2024_v3')
CORS(app)

# ── MediaPipe ────────────────────────────────────────────────────────────────
mp_pose    = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose_detector = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# ── Geometry helpers ─────────────────────────────────────────────────────────
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cos_a = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return math.degrees(math.acos(np.clip(cos_a, -1.0, 1.0)))

def lm_norm(landmarks, idx):
    lm = landmarks[idx]
    return [lm.x, lm.y]

# ── Global state ─────────────────────────────────────────────────────────────
exercise_state = {}
# In-memory dashboard stats per session
dashboard_stats = {}

def get_state(sid, ex):
    key = f"{sid}_{ex}"
    if key not in exercise_state:
        exercise_state[key] = {
            'stage': None, 'reps': 0, 'correct_reps': 0, 'wrong_reps': 0,
            'mistakes': [], 'last_angle': 0, 'start_time': time.time(),
            'plank_start': None, 'jj_stage': None, 'hk_stage': None,
            'form_accuracy_sum': 0, 'form_checks': 0,
        }
    return exercise_state[key]

def get_dashboard(sid):
    if sid not in dashboard_stats:
        dashboard_stats[sid] = {
            'workouts_completed': 0,
            'total_reps': 0,
            'accuracy_sum': 0,
            'accuracy_count': 0,
            'exercise_history': [],   # list of {exercise, reps, accuracy, ts}
        }
    return dashboard_stats[sid]

# ═══════════════════════════════════════════════════════════════════════════
#  EXERCISE DETECTORS — all return (feedback, form_ok, accuracy 0-100)
# ═══════════════════════════════════════════════════════════════════════════

def detect_squat(lm, st):
    hip   = lm_norm(lm, 23)
    knee  = lm_norm(lm, 25)
    ankle = lm_norm(lm, 27)
    ang   = calculate_angle(hip, knee, ankle)
    st['last_angle'] = round(ang, 1)
    feedback, ok = "Stand with feet shoulder-width apart.", True

    if ang < 90:
        if st['stage'] == 'up':
            st['stage'] = 'down'
        feedback = "Great depth! Drive back up through heels."
    elif ang > 160:
        if st['stage'] == 'down':
            st['reps'] += 1
            st['correct_reps'] += 1
            st['stage'] = 'up'
            feedback = f"Rep {st['reps']} complete! Perfect squat."
        else:
            st['stage'] = 'up'
            feedback = "Lower your hips – go deeper than parallel."
    else:
        feedback = "Keep lowering – get below parallel."

    if ang < 120:
        shoulder = lm_norm(lm, 11)
        if abs(shoulder[0] - hip[0]) > 0.14:
            feedback = "Back leaning forward – keep your chest up!"
            ok = False
            if "Back leaning forward" not in st['mistakes']:
                st['mistakes'].append("Back leaning forward")
        lknee = lm_norm(lm, 25); rknee = lm_norm(lm, 26); rhip = lm_norm(lm, 24)
        if (lknee[0] - hip[0]) > 0.07 or (rhip[0] - rknee[0]) > 0.07:
            feedback = "Knees moving inward – push your knees outward!"
            ok = False
            if "Knees caving inward" not in st['mistakes']:
                st['mistakes'].append("Knees caving inward")
    if ang > 100 and ang < 150 and st['stage'] == 'down':
        feedback = "Not deep enough – lower your hips more!"
        ok = False

    acc = max(0, min(100, int((1 - (ang - 90) / 70) * 100))) if ang < 160 else 0
    return feedback, ok, acc

def detect_pushup(lm, st):
    sh  = lm_norm(lm, 11); el = lm_norm(lm, 13); wr = lm_norm(lm, 15)
    ang = calculate_angle(sh, el, wr)
    st['last_angle'] = round(ang, 1)
    feedback, ok = "Get into plank – hands under shoulders.", True

    if ang < 90:
        if st['stage'] == 'up':
            st['stage'] = 'down'
        feedback = "Good depth! Push back up powerfully."
    elif ang > 160:
        if st['stage'] == 'down':
            st['reps'] += 1
            st['correct_reps'] += 1
            st['stage'] = 'up'
            feedback = f"Rep {st['reps']} complete! Lower again."
        else:
            st['stage'] = 'up'
            feedback = "Lower your chest toward the floor."
    elif 90 < ang < 140 and st['stage'] == 'up':
        feedback = "Half push-up detected – go lower!"
        ok = False
        if "Half push-up" not in st['mistakes']:
            st['mistakes'].append("Half push-up")

    lsh = lm_norm(lm, 11); rsh = lm_norm(lm, 12)
    lel = lm_norm(lm, 13); rel = lm_norm(lm, 14)
    sw = abs(rsh[0] - lsh[0]); ew = abs(rel[0] - lel[0])
    if ew > sw * 1.45:
        feedback = "Elbows too wide – keep elbows closer to body!"
        ok = False
        if "Elbows too wide" not in st['mistakes']:
            st['mistakes'].append("Elbows too wide")

    hip = lm_norm(lm, 23); ank = lm_norm(lm, 27)
    body_ang = calculate_angle(sh, hip, ank)
    if body_ang < 155:
        feedback = "Hips sagging – keep body in a straight line!"
        ok = False
        if "Hips sagging" not in st['mistakes']:
            st['mistakes'].append("Hips sagging")

    acc = max(0, min(100, int((1 - ang / 90) * 100))) if ang < 90 else 0
    return feedback, ok, acc

def detect_bicep_curl(lm, st):
    sh  = lm_norm(lm, 11); el = lm_norm(lm, 13); wr = lm_norm(lm, 15)
    ang = calculate_angle(sh, el, wr)
    st['last_angle'] = round(ang, 1)
    feedback, ok = "Extend arm fully, palm facing up.", True

    if ang < 50:
        if st['stage'] == 'down':
            st['stage'] = 'up'
        feedback = "Good curl! Lower slowly and controlled."
    elif ang > 160:
        if st['stage'] == 'up':
            st['reps'] += 1
            st['correct_reps'] += 1
            st['stage'] = 'down'
            feedback = f"Rep {st['reps']} complete! Full range."
        else:
            st['stage'] = 'down'
            feedback = "Curl your arm all the way up."
    else:
        feedback = "Keep curling – bring wrist to shoulder height."

    if abs(el[0] - sh[0]) > 0.10:
        feedback = "Elbow drifting – keep elbow pinned to your side!"
        ok = False
        if "Elbow drifting" not in st['mistakes']:
            st['mistakes'].append("Elbow drifting")

    acc = max(0, min(100, int((160 - ang) / 110 * 100))) if ang < 160 else 0
    return feedback, ok, acc

def detect_lunge(lm, st):
    hip  = lm_norm(lm, 23); knee = lm_norm(lm, 25); ankle = lm_norm(lm, 27)
    ang  = calculate_angle(hip, knee, ankle)
    st['last_angle'] = round(ang, 1)
    feedback, ok = "Step forward – feet hip-width apart.", True

    if ang < 95:
        if st['stage'] == 'up':
            st['stage'] = 'down'
        feedback = "Great lunge depth! Drive back through heel."
    elif ang > 165:
        if st['stage'] == 'down':
            st['reps'] += 1
            st['correct_reps'] += 1
            st['stage'] = 'up'
            feedback = f"Rep {st['reps']} complete! Switch legs."
        else:
            st['stage'] = 'up'
            feedback = "Step forward and lower rear knee toward floor."

    shoulder = lm_norm(lm, 11)
    if abs(shoulder[0] - hip[0]) > 0.12:
        feedback = "Back leaning forward – keep your chest up!"
        ok = False
        if "Torso leaning" not in st['mistakes']:
            st['mistakes'].append("Torso leaning")

    acc = max(0, min(100, int((165 - ang) / 70 * 100))) if ang < 165 else 0
    return feedback, ok, acc

def detect_shoulder_press(lm, st):
    sh  = lm_norm(lm, 11); el = lm_norm(lm, 13); wr = lm_norm(lm, 15)
    ang = calculate_angle(sh, el, wr)
    st['last_angle'] = round(ang, 1)
    feedback, ok = "Hold weights at shoulders, core braced.", True

    if ang > 160:
        if st['stage'] == 'down':
            st['stage'] = 'up'
        feedback = "Full lockout! Lower with control."
    elif ang < 80:
        if st['stage'] == 'up':
            st['reps'] += 1
            st['correct_reps'] += 1
            st['stage'] = 'down'
            feedback = f"Rep {st['reps']} complete! Press again."
        else:
            st['stage'] = 'down'
            feedback = "Press weights directly overhead."

    hip = lm_norm(lm, 23)
    if abs(sh[0] - hip[0]) > 0.10:
        feedback = "Don't arch your back – brace your core!"
        ok = False
        if "Lower back arch" not in st['mistakes']:
            st['mistakes'].append("Lower back arch")

    acc = max(0, min(100, int((ang - 80) / 80 * 100))) if ang > 80 else 100
    return feedback, ok, acc

def detect_plank(lm, st):
    sh  = lm_norm(lm, 11); hip = lm_norm(lm, 23); ank = lm_norm(lm, 27)
    ang = calculate_angle(sh, hip, ank)
    st['last_angle'] = round(ang, 1)
    ok = True

    if 155 < ang < 205:
        if st['plank_start'] is None:
            st['plank_start'] = time.time()
        dur = int(time.time() - st['plank_start'])
        st['reps'] = dur
        st['correct_reps'] = dur // 10
        feedback = f"Solid plank! Hold it – {dur}s elapsed."
    elif ang <= 155:
        st['plank_start'] = None
        feedback = "Hips too high – lower them to a straight line!"
        ok = False
        if "Hips too high" not in st['mistakes']:
            st['mistakes'].append("Hips too high")
    else:
        st['plank_start'] = None
        feedback = "Hips sagging – squeeze glutes and raise hips!"
        ok = False
        if "Hips sagging" not in st['mistakes']:
            st['mistakes'].append("Hips sagging")

    acc = max(0, min(100, int((1 - abs(180 - ang) / 25) * 100))) if 155 < ang < 205 else 0
    return feedback, ok, acc

def detect_jumping_jacks(lm, st):
    lw = lm_norm(lm, 15); rw = lm_norm(lm, 16)
    la = lm_norm(lm, 27); ra = lm_norm(lm, 28)
    arm_sp = abs(rw[0] - lw[0]); leg_sp = abs(ra[0] - la[0])
    st['last_angle'] = round(arm_sp * 100, 1)
    ok = True

    if arm_sp > 0.50 and leg_sp > 0.20:
        if st['jj_stage'] == 'closed':
            st['reps'] += 1
            st['correct_reps'] += 1
            feedback = f"Rep {st['reps']}! Keep the rhythm."
        else:
            feedback = "Arms and legs wide! Good form."
        st['jj_stage'] = 'open'
    elif arm_sp < 0.25:
        if st['jj_stage'] == 'open':
            st['reps'] += 1
            st['correct_reps'] += 1
            feedback = f"Rep {st['reps']}! Jump wide again."
        else:
            feedback = "Jump – spread arms and legs wide!"
        st['jj_stage'] = 'closed'
    else:
        feedback = "Spread your arms and legs wider!"

    acc = max(0, min(100, int(arm_sp / 0.50 * 100)))
    return feedback, ok, acc

def detect_high_knees(lm, st):
    lh = lm_norm(lm, 23); rh = lm_norm(lm, 24)
    lk = lm_norm(lm, 25); rk = lm_norm(lm, 26)
    l_raise = lh[1] - lk[1]; r_raise = rh[1] - rk[1]
    st['last_angle'] = round(max(l_raise, r_raise) * 100, 1)
    ok = True

    if l_raise > 0.06 or r_raise > 0.06:
        if st['hk_stage'] == 'down':
            st['reps'] += 1
            st['correct_reps'] += 1
            feedback = f"Rep {st['reps']}! Drive those knees high!"
        else:
            feedback = "Knee up! Great high knees!"
        st['hk_stage'] = 'up'
    else:
        if st['hk_stage'] == 'up':
            st['hk_stage'] = 'down'
        feedback = "Raise your knees higher – above hip level!"
        if st['reps'] > 0:
            ok = False
            if "Knees not high enough" not in st['mistakes']:
                st['mistakes'].append("Knees not high enough")

    acc = max(0, min(100, int(max(l_raise, r_raise) / 0.06 * 100)))
    return feedback, ok, acc

def detect_arm_raises(lm, st):
    sh = lm_norm(lm, 11); wr = lm_norm(lm, 15)
    height = sh[1] - wr[1]
    st['last_angle'] = round(height * 100, 1)
    ok = True

    if height > 0.12:
        if st['stage'] == 'down':
            st['stage'] = 'up'
        feedback = "Arms raised! Lower with control."
    elif height < -0.05:
        if st['stage'] == 'up':
            st['reps'] += 1
            st['correct_reps'] += 1
            st['stage'] = 'down'
            feedback = f"Rep {st['reps']} complete! Raise again."
        else:
            st['stage'] = 'down'
            feedback = "Raise arms to shoulder height."
    else:
        feedback = "Lift arms parallel to the floor."

    acc = max(0, min(100, int(height / 0.12 * 100)))
    return feedback, ok, acc

def detect_side_lunge(lm, st):
    hip  = lm_norm(lm, 23); knee = lm_norm(lm, 25); ankle = lm_norm(lm, 27)
    ang  = calculate_angle(hip, knee, ankle)
    st['last_angle'] = round(ang, 1)
    ok = True

    if ang < 100:
        if st['stage'] == 'up':
            st['stage'] = 'down'
        feedback = "Deep side lunge! Push back up."
    elif ang > 165:
        if st['stage'] == 'down':
            st['reps'] += 1
            st['correct_reps'] += 1
            st['stage'] = 'up'
            feedback = f"Rep {st['reps']} complete! Step wide again."
        else:
            st['stage'] = 'up'
            feedback = "Step wide to the side and bend your knee."
    else:
        feedback = "Sink lower – get knee to 90 degrees."

    if abs(knee[0] - ankle[0]) > 0.10:
        feedback = "Knee passing over toes – align knee over foot!"
        ok = False
        if "Knee over toes" not in st['mistakes']:
            st['mistakes'].append("Knee over toes")

    acc = max(0, min(100, int((165 - ang) / 65 * 100))) if ang < 165 else 0
    return feedback, ok, acc

DETECTORS = {
    'squat': detect_squat, 'pushup': detect_pushup, 'bicep_curl': detect_bicep_curl,
    'lunge': detect_lunge, 'shoulder_press': detect_shoulder_press, 'plank': detect_plank,
    'jumping_jacks': detect_jumping_jacks, 'high_knees': detect_high_knees,
    'arm_raises': detect_arm_raises, 'side_lunge': detect_side_lunge,
}

# ═══════════════════════════════════════════════════════════════════════════
#  SKELETON & HUD DRAWING
# ═══════════════════════════════════════════════════════════════════════════

def draw_skeleton(frame, pose_landmarks, form_ok=True):
    h, w = frame.shape[:2]
    conn_color  = (50, 220, 120) if form_ok else (60, 100, 255)
    point_color = (0, 200, 255)
    point_bad   = (50, 50, 255)

    CONNECTIONS = [
        (11,12),(11,23),(12,24),(23,24),
        (11,13),(13,15),(12,14),(14,16),
        (23,25),(25,27),(27,29),(29,31),
        (24,26),(26,28),(28,30),(30,32),
    ]
    KEY_JOINTS = [11,12,13,14,15,16,23,24,25,26,27,28]
    lm = pose_landmarks.landmark

    for a, b in CONNECTIONS:
        lmA, lmB = lm[a], lm[b]
        if lmA.visibility < 0.35 or lmB.visibility < 0.35:
            continue
        pt1 = (int(lmA.x * w), int(lmA.y * h))
        pt2 = (int(lmB.x * w), int(lmB.y * h))
        cv2.line(frame, pt1, pt2, conn_color, 3, cv2.LINE_AA)

    for idx in KEY_JOINTS:
        lmk = lm[idx]
        if lmk.visibility < 0.35:
            continue
        x, y = int(lmk.x * w), int(lmk.y * h)
        c = point_bad if not form_ok else point_color
        cv2.circle(frame, (x, y), 7, c, -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 9, (255,255,255), 2, cv2.LINE_AA)

    return frame

def add_hud(frame, exercise, reps, angle, form_ok, accuracy, feedback):
    h, w = frame.shape[:2]
    ov = frame.copy()
    cv2.rectangle(ov, (0,0), (w,52), (8,10,18), -1)
    cv2.addWeighted(ov, 0.68, frame, 0.32, 0, frame)

    ex_label = exercise.replace('_',' ').upper()
    cv2.putText(frame, ex_label, (12,36), cv2.FONT_HERSHEY_DUPLEX, 0.90, (0,200,255), 2, cv2.LINE_AA)

    rep_str = f"REPS: {reps}"
    cv2.putText(frame, rep_str, (w//2-65,36), cv2.FONT_HERSHEY_DUPLEX, 0.95, (255,255,255), 2, cv2.LINE_AA)

    acc_color = (50,220,120) if accuracy>70 else (50,180,255) if accuracy>40 else (60,60,255)
    cv2.putText(frame, f"ACC:{accuracy}%", (w-148,36), cv2.FONT_HERSHEY_DUPLEX, 0.80, acc_color, 2, cv2.LINE_AA)

    ov2 = frame.copy()
    by = h-58
    cv2.rectangle(ov2, (0,by), (w,h), (8,10,18), -1)
    cv2.addWeighted(ov2, 0.72, frame, 0.28, 0, frame)

    form_color = (50,220,120) if form_ok else (60,100,255)
    cv2.putText(frame, "FORM OK" if form_ok else "FIX FORM", (12,by+22), cv2.FONT_HERSHEY_DUPLEX, 0.60, form_color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"Angle:{angle}", (12,by+44), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (160,160,160), 1, cv2.LINE_AA)
    fb_short = feedback[:62]+('…' if len(feedback)>62 else '')
    cv2.putText(frame, fb_short, (130,by+34), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (225,225,225), 1, cv2.LINE_AA)

    return frame

# ═══════════════════════════════════════════════════════════════════════════
#  DAILY PLAN GENERATION
# ═══════════════════════════════════════════════════════════════════════════

def generate_daily_plan(profile):
    goal   = profile.get('fitness_goal', 'general_fitness')
    weight = float(profile.get('weight', 70))
    age    = int(profile.get('age', 25))

    # Determine intensity based on age
    intensity = 'beginner' if age > 50 else 'intermediate' if age > 35 else 'advanced'

    plans = {
        'muscle_gain': {
            'label': 'Muscle Gain Plan',
            'warmup':   [{'name':'Jumping Jacks','sets':2,'reps':'15','rest':'30s'}],
            'workout':  [
                {'name':'Squat',         'sets':3,'reps':'12','rest':'60s','key':'squat'},
                {'name':'Push-up',       'sets':3,'reps':'10','rest':'60s','key':'pushup'},
                {'name':'Shoulder Press','sets':3,'reps':'10','rest':'60s','key':'shoulder_press'},
                {'name':'Bicep Curl',    'sets':3,'reps':'12','rest':'45s','key':'bicep_curl'},
                {'name':'Lunge',         'sets':3,'reps':'10','rest':'60s','key':'lunge'},
            ],
            'cooldown': [{'name':'Full Body Stretch','duration':'5 min'}],
        },
        'weight_loss': {
            'label': 'Fat Burn Plan',
            'warmup':   [{'name':'High Knees','sets':1,'reps':'30s','rest':'15s'}],
            'workout':  [
                {'name':'Jumping Jacks','sets':3,'reps':'20',  'rest':'30s','key':'jumping_jacks'},
                {'name':'High Knees',   'sets':3,'reps':'30s', 'rest':'30s','key':'high_knees'},
                {'name':'Lunge',        'sets':3,'reps':'12',  'rest':'45s','key':'lunge'},
                {'name':'Squat',        'sets':3,'reps':'15',  'rest':'45s','key':'squat'},
                {'name':'Plank',        'sets':3,'reps':'30s', 'rest':'30s','key':'plank'},
            ],
            'cooldown': [{'name':'Stretching & Deep Breathing','duration':'5 min'}],
        },
        'body_recomposition': {
            'label': 'Recomposition Plan',
            'warmup':   [{'name':'Jumping Jacks','sets':2,'reps':'15','rest':'20s'}],
            'workout':  [
                {'name':'Squat',        'sets':3,'reps':'12','rest':'45s','key':'squat'},
                {'name':'Push-up',      'sets':3,'reps':'10','rest':'45s','key':'pushup'},
                {'name':'Bicep Curl',   'sets':3,'reps':'12','rest':'45s','key':'bicep_curl'},
                {'name':'Jumping Jacks','sets':3,'reps':'20','rest':'30s','key':'jumping_jacks'},
                {'name':'Plank',        'sets':3,'reps':'30s','rest':'30s','key':'plank'},
                {'name':'Side Lunge',   'sets':2,'reps':'10','rest':'45s','key':'side_lunge'},
            ],
            'cooldown': [{'name':'Yoga Stretching','duration':'5 min'}],
        },
        'general_fitness': {
            'label': 'General Fitness Plan',
            'warmup':   [{'name':'Arm Raises','sets':2,'reps':'12','rest':'20s'}],
            'workout':  [
                {'name':'Squat',     'sets':3,'reps':'12','rest':'45s','key':'squat'},
                {'name':'Lunge',     'sets':2,'reps':'10','rest':'45s','key':'lunge'},
                {'name':'Push-up',   'sets':3,'reps':'8', 'rest':'45s','key':'pushup'},
                {'name':'Arm Raises','sets':3,'reps':'12','rest':'30s','key':'arm_raises'},
                {'name':'Plank',     'sets':3,'reps':'20s','rest':'30s','key':'plank'},
            ],
            'cooldown': [{'name':'Light Stretching','duration':'5 min'}],
        },
    }

    plan = plans.get(goal, plans['general_fitness'])
    # Adjust for beginner
    if intensity == 'beginner':
        for ex in plan['workout']:
            ex['sets'] = max(2, ex['sets'] - 1)

    return plan

# ═══════════════════════════════════════════════════════════════════════════
#  NUTRITION / DIET
# ═══════════════════════════════════════════════════════════════════════════

GOAL_MEALS = {
    'weight_loss': {
        'breakfast': 'Oats (50g) + Milk (200ml) + 2 Egg whites',
        'lunch':     'Brown Rice (100g) + Grilled Chicken (150g) + Salad',
        'snack':     'Greek Yogurt (150g) + Apple',
        'dinner':    'Chapati (2) + Dal (1 cup) + Paneer (100g)',
        'pre_bed':   'Low-fat Milk (200ml)',
        'foods':     ['Chicken Breast','Fish','Egg whites','Oats','Greek Yogurt','Vegetables','Lentils'],
    },
    'muscle_gain': {
        'breakfast': 'Oats (80g) + Whole Milk (300ml) + 3 Eggs',
        'lunch':     'Rice (200g) + Chicken Breast (200g) + Vegetables',
        'snack':     'Peanut Butter Sandwich + Banana',
        'dinner':    'Chapati (3) + Dal (1.5 cups) + 2 Eggs',
        'pre_bed':   'Full-fat Milk (300ml) + Peanut Butter (30g)',
        'foods':     ['Chicken Breast','Eggs','Paneer','Fish','Milk','Peanut Butter','Tofu','Lentils'],
    },
    'body_recomposition': {
        'breakfast': 'Oats (60g) + Milk (250ml) + 2 Eggs',
        'lunch':     'Rice (150g) + Chicken or Paneer (150g) + Vegetables',
        'snack':     'Greek Yogurt (150g) + Mixed Nuts (30g)',
        'dinner':    'Chapati (2) + Dal (1 cup) + Fish (150g)',
        'pre_bed':   'Milk (250ml)',
        'foods':     ['Chicken Breast','Fish','Eggs','Paneer','Greek Yogurt','Tofu','Peanut Butter','Lentils'],
    },
    'general_fitness': {
        'breakfast': 'Oats (50g) + Milk (200ml) + 2 Eggs',
        'lunch':     'Rice (150g) + Chicken or Paneer (120g) + Vegetables',
        'snack':     'Peanut Butter Sandwich',
        'dinner':    'Chapati (2) + Dal (1 cup) + 1 Egg',
        'pre_bed':   'Milk (200ml)',
        'foods':     ['Eggs','Chicken Breast','Fish','Paneer','Milk','Greek Yogurt','Tofu','Peanut Butter','Lentils'],
    },
}

GOAL_RECS = {
    'weight_loss':        ['Jumping Jacks','High Knees','Squat','Lunge','Push-up'],
    'muscle_gain':        ['Squat','Push-up','Bicep Curl','Shoulder Press','Side Lunge'],
    'body_recomposition': ['Squat','Push-up','Bicep Curl','Jumping Jacks','Plank'],
    'general_fitness':    ['Squat','Lunge','Push-up','Arm Raises','Plank'],
}

def build_diet(profile):
    w    = float(profile.get('weight', 70))
    goal = profile.get('fitness_goal', 'general_fitness')
    pl   = round(w * 1.6, 1); ph = round(w * 2.2, 1)
    meals = GOAL_MEALS.get(goal, GOAL_MEALS['general_fitness'])
    return {
        'protein_range': f'{pl}g – {ph}g',
        'protein_low': pl, 'protein_high': ph,
        'meals': meals,
        'goal': goal.replace('_',' ').title(),
    }

# ═══════════════════════════════════════════════════════════════════════════
#  PDF GENERATION
# ═══════════════════════════════════════════════════════════════════════════

def generate_nutrition_pdf(profile):
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm)

    styles = getSampleStyleSheet()
    W = A4[0] - 4*cm

    title_s = ParagraphStyle('t', parent=styles['Title'],
        fontSize=22, textColor=colors.HexColor('#0066ff'),
        spaceAfter=4, fontName='Helvetica-Bold', alignment=TA_CENTER)
    sub_s   = ParagraphStyle('s', parent=styles['Normal'],
        fontSize=11, textColor=colors.HexColor('#5a5f7a'),
        spaceAfter=4, alignment=TA_CENTER)
    sec_s   = ParagraphStyle('sec', parent=styles['Heading2'],
        fontSize=13, textColor=colors.HexColor('#0f1117'),
        fontName='Helvetica-Bold', spaceBefore=14, spaceAfter=6)
    body_s  = ParagraphStyle('b', parent=styles['Normal'],
        fontSize=10, textColor=colors.HexColor('#2a2d3a'), spaceAfter=4, leading=15)
    hl_s    = ParagraphStyle('hl', parent=styles['Normal'],
        fontSize=14, textColor=colors.HexColor('#0066ff'),
        fontName='Helvetica-Bold', spaceAfter=4, alignment=TA_CENTER)

    diet   = build_diet(profile)
    plan   = generate_daily_plan(profile)
    name   = profile.get('name','User')
    age    = profile.get('age','—')
    height = profile.get('height','—')
    weight = profile.get('weight','—')
    gender = profile.get('gender','—').title()
    goal   = diet['goal']
    pl, ph = diet['protein_low'], diet['protein_high']
    meals  = diet['meals']
    story  = []

    story.append(Paragraph("⚡ FitAI", title_s))
    story.append(Paragraph("AI Workout Trainer – Personal Nutrition & Workout Plan", sub_s))
    story.append(HRFlowable(width=W, thickness=2, color=colors.HexColor('#0066ff'), spaceAfter=12))

    story.append(Paragraph("Personal Information", sec_s))
    user_data = [
        ['Name',name,'Age',str(age)],
        ['Height',f'{height} cm','Weight',f'{weight} kg'],
        ['Gender',gender,'Goal',goal],
    ]
    t = Table(user_data, colWidths=[W*.22,W*.28,W*.22,W*.28])
    t.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,-1),colors.HexColor('#f0f1f5')),
        ('BACKGROUND',(0,0),(0,-1),colors.HexColor('#e8eeff')),
        ('BACKGROUND',(2,0),(2,-1),colors.HexColor('#e8eeff')),
        ('FONTNAME',(0,0),(0,-1),'Helvetica-Bold'),
        ('FONTNAME',(2,0),(2,-1),'Helvetica-Bold'),
        ('FONTSIZE',(0,0),(-1,-1),10),
        ('TEXTCOLOR',(0,0),(0,-1),colors.HexColor('#0066ff')),
        ('TEXTCOLOR',(2,0),(2,-1),colors.HexColor('#0066ff')),
        ('GRID',(0,0),(-1,-1),.5,colors.HexColor('#e4e6ec')),
        ('PADDING',(0,0),(-1,-1),8),
        ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
    ]))
    story.append(t)
    story.append(Spacer(1,10))

    story.append(HRFlowable(width=W,thickness=1,color=colors.HexColor('#e4e6ec'),spaceAfter=6))
    story.append(Paragraph("Daily Protein Requirement", sec_s))
    story.append(Paragraph(f"Formula: Body Weight × 1.6g to 2.2g per kg  →  <b>{weight} kg × 1.6–2.2 = {pl}g – {ph}g/day</b>", body_s))
    prot_box = Table([[Paragraph(f"{pl}g – {ph}g per day", hl_s)]],colWidths=[W])
    prot_box.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,-1),colors.HexColor('#e8eeff')),
        ('PADDING',(0,0),(-1,-1),12),
        ('BOX',(0,0),(-1,-1),1.5,colors.HexColor('#0066ff')),
    ]))
    story.append(prot_box)
    story.append(Spacer(1,10))

    story.append(HRFlowable(width=W,thickness=1,color=colors.HexColor('#e4e6ec'),spaceAfter=6))
    story.append(Paragraph("Daily Meal Plan", sec_s))
    meal_rows = [[Paragraph('<b>Meal</b>',body_s),Paragraph('<b>Food</b>',body_s)]]
    for label, content in [
        ('🌅 Breakfast',meals['breakfast']),('☀️ Lunch',meals['lunch']),
        ('🍎 Snack',meals['snack']),('🌙 Dinner',meals['dinner']),
        ('🌛 Before Bed',meals['pre_bed']),
    ]:
        meal_rows.append([Paragraph(label,body_s),Paragraph(content,body_s)])
    mt = Table(meal_rows, colWidths=[W*.28,W*.72])
    mt.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#0066ff')),
        ('TEXTCOLOR',(0,0),(-1,0),colors.white),
        ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
        ('FONTSIZE',(0,0),(-1,-1),10),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#f7f8fa'),colors.white]),
        ('GRID',(0,0),(-1,-1),.5,colors.HexColor('#e4e6ec')),
        ('PADDING',(0,0),(-1,-1),9),
        ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
    ]))
    story.append(mt)
    story.append(Spacer(1,10))

    story.append(HRFlowable(width=W,thickness=1,color=colors.HexColor('#e4e6ec'),spaceAfter=6))
    story.append(Paragraph(f"Workout Plan: {plan['label']}", sec_s))

    wp_rows = [[Paragraph('<b>Exercise</b>',body_s),Paragraph('<b>Sets</b>',body_s),
                Paragraph('<b>Reps / Duration</b>',body_s),Paragraph('<b>Rest</b>',body_s)]]
    for ex in plan['workout']:
        wp_rows.append([
            Paragraph(ex['name'],body_s),
            Paragraph(str(ex['sets']),body_s),
            Paragraph(str(ex['reps']),body_s),
            Paragraph(ex.get('rest','30s'),body_s),
        ])
    wt = Table(wp_rows, colWidths=[W*.40,W*.15,W*.25,W*.20])
    wt.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#0066ff')),
        ('TEXTCOLOR',(0,0),(-1,0),colors.white),
        ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
        ('FONTSIZE',(0,0),(-1,-1),10),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#f7f8fa'),colors.white]),
        ('GRID',(0,0),(-1,-1),.5,colors.HexColor('#e4e6ec')),
        ('PADDING',(0,0),(-1,-1),9),
        ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
    ]))
    story.append(wt)
    story.append(Spacer(1,10))

    story.append(HRFlowable(width=W,thickness=1,color=colors.HexColor('#e4e6ec'),spaceAfter=6))
    story.append(Paragraph("Recommended High-Protein Foods", sec_s))
    food_list = meals['foods']
    rows_f = [food_list[i:i+3] for i in range(0,len(food_list),3)]
    while rows_f and len(rows_f[-1])<3: rows_f[-1].append('')
    ft = Table([[Paragraph(f"✔ {f}",body_s) if f else Paragraph('',body_s) for f in row] for row in rows_f], colWidths=[W/3]*3)
    ft.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,-1),colors.HexColor('#f0fff8')),
        ('GRID',(0,0),(-1,-1),.5,colors.HexColor('#d0f0e0')),
        ('PADDING',(0,0),(-1,-1),8),
        ('TEXTCOLOR',(0,0),(-1,-1),colors.HexColor('#007a50')),
    ]))
    story.append(ft)
    story.append(Spacer(1,10))

    story.append(HRFlowable(width=W,thickness=1,color=colors.HexColor('#e4e6ec'),spaceAfter=6))
    footer_s = ParagraphStyle('f',parent=styles['Normal'],fontSize=8,
        textColor=colors.HexColor('#9499b5'),alignment=TA_CENTER)
    story.append(Paragraph("Generated by FitAI – AI Personal Trainer Platform  |  fitai.app", footer_s))

    doc.build(story)
    buf.seek(0)
    return buf

# ═══════════════════════════════════════════════════════════════════════════
#  FLASK ROUTES
# ═══════════════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/profile')
def profile():
    return render_template('profile.html')

@app.route('/api/process_frame', methods=['POST'])
def process_frame():
    try:
        data       = request.json
        frame_b64  = data.get('frame','')
        exercise   = data.get('exercise','squat')
        session_id = data.get('session_id','default')

        if ',' in frame_b64:
            frame_b64 = frame_b64.split(',')[1]
        img_bytes = base64.b64decode(frame_b64)
        img_arr   = np.frombuffer(img_bytes, dtype=np.uint8)
        frame     = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({'error':'Invalid frame'}), 400

        proc_h = 480
        proc_w = int(frame.shape[1] * proc_h / frame.shape[0])
        frame  = cv2.resize(frame, (proc_w, proc_h))

        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose_detector.process(rgb)

        if not results.pose_landmarks:
            cv2.putText(frame, "Position yourself – full body visible",
                (20, frame.shape[0]//2), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (200,200,200), 2, cv2.LINE_AA)
            _, enc = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 78])
            return jsonify({
                'detected': False,
                'feedback': 'Position yourself so your full body is visible.',
                'reps':0,'angle':0,'form_ok':False,'accuracy':0,
                'correct_reps':0,'wrong_reps':0,'mistakes':[],
                'processed_frame': 'data:image/jpeg;base64,'+base64.b64encode(enc).decode(),
            })

        lm  = results.pose_landmarks.landmark
        st  = get_state(session_id, exercise)
        fn  = DETECTORS.get(exercise, detect_squat)
        feedback, ok, accuracy = fn(lm, st)

        # Track accuracy for dashboard
        st['form_accuracy_sum'] += accuracy
        st['form_checks'] += 1

        # Update dashboard stats
        dash = get_dashboard(session_id)
        dash['total_reps'] = sum(
            exercise_state[k]['reps']
            for k in exercise_state
            if k.startswith(session_id)
        )
        if accuracy > 0:
            dash['accuracy_sum'] += accuracy
            dash['accuracy_count'] += 1

        frame = draw_skeleton(frame, results.pose_landmarks, ok)
        frame = add_hud(frame, exercise, st['reps'], st['last_angle'], ok, accuracy, feedback)

        _, enc = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
        frame_out = 'data:image/jpeg;base64,' + base64.b64encode(enc).decode()

        return jsonify({
            'detected': True,
            'feedback': feedback,
            'reps': st['reps'],
            'angle': st['last_angle'],
            'form_ok': ok,
            'accuracy': accuracy,
            'correct_reps': st['correct_reps'],
            'wrong_reps': st['wrong_reps'],
            'mistakes': st['mistakes'][-3:],
            'processed_frame': frame_out,
        })

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/reset_exercise', methods=['POST'])
def reset_exercise():
    data = request.json
    key  = f"{data.get('session_id','default')}_{data.get('exercise','squat')}"
    if key in exercise_state:
        del exercise_state[key]
    return jsonify({'status':'reset'})

@app.route('/api/get_summary', methods=['POST'])
def get_summary():
    data = request.json
    key  = f"{data.get('session_id','default')}_{data.get('exercise','squat')}"
    st   = exercise_state.get(key, {})
    reps = st.get('reps', 0)
    corr = st.get('correct_reps', 0)
    wrng = st.get('wrong_reps', 0)
    acc  = round(corr / reps * 100, 1) if reps > 0 else 0
    return jsonify({
        'exercise': data.get('exercise','squat'),
        'total_reps': reps,'correct_reps': corr,'wrong_reps': wrng,
        'accuracy': acc,'mistakes': list(set(st.get('mistakes',[]))),
    })

@app.route('/api/save_profile', methods=['POST'])
def save_profile():
    p    = request.json
    session['profile'] = p
    diet = build_diet(p)
    plan = generate_daily_plan(p)
    recs = GOAL_RECS.get(p.get('fitness_goal','general_fitness'), GOAL_RECS['general_fitness'])
    return jsonify({'status':'saved','diet':diet,'plan':plan,'recommendations':recs})

@app.route('/api/get_profile')
def get_profile():
    p = session.get('profile', {})
    if not p:
        return jsonify({'profile': None})
    diet = build_diet(p)
    plan = generate_daily_plan(p)
    recs = GOAL_RECS.get(p.get('fitness_goal','general_fitness'), GOAL_RECS['general_fitness'])
    return jsonify({'profile':p,'diet':diet,'plan':plan,'recommendations':recs})

@app.route('/api/get_daily_plan', methods=['POST'])
def get_daily_plan():
    profile = request.json or session.get('profile', {})
    if not profile:
        return jsonify({'error':'No profile'}), 400
    plan = generate_daily_plan(profile)
    diet = build_diet(profile)
    return jsonify({'plan': plan, 'diet': diet})

@app.route('/api/dashboard_stats', methods=['GET'])
def dashboard_stats_route():
    sid  = request.args.get('session_id', 'default')
    dash = get_dashboard(sid)

    # Build per-exercise breakdown
    exercises_done = []
    for key, st in exercise_state.items():
        if not key.startswith(sid): continue
        parts = key.split('_', 1)
        if len(parts) < 2: continue
        ex    = parts[1]
        reps  = st.get('reps', 0)
        corr  = st.get('correct_reps', 0)
        checks = st.get('form_checks', 1) or 1
        avg_acc = round(st.get('form_accuracy_sum', 0) / checks, 1)
        if reps > 0:
            exercises_done.append({
                'exercise': ex.replace('_',' ').title(),
                'reps': reps,
                'correct': corr,
                'accuracy': avg_acc,
                'mistakes': list(set(st.get('mistakes', []))),
            })

    total_reps  = sum(e['reps'] for e in exercises_done)
    overall_acc = round(dash['accuracy_sum'] / dash['accuracy_count'], 1) if dash['accuracy_count'] else 0
    completed   = len([e for e in exercises_done if e['reps'] >= 5])

    return jsonify({
        'workouts_completed': completed,
        'total_reps':         total_reps,
        'overall_accuracy':   overall_acc,
        'exercises':          exercises_done,
    })

@app.route('/api/workout_history')
def workout_history():
    history = []
    for key, st in exercise_state.items():
        parts = key.split('_', 1)
        if len(parts) == 2 and st.get('reps', 0) > 0:
            reps = st['reps']; corr = st['correct_reps']
            checks = st.get('form_checks', 1) or 1
            history.append({
                'exercise': parts[1],
                'reps': reps,
                'accuracy': round(st.get('form_accuracy_sum',0)/checks, 1),
                'mistakes': list(set(st.get('mistakes',[])))
            })
    return jsonify({'history': history})

@app.route('/api/download_nutrition_pdf', methods=['POST'])
def download_nutrition_pdf():
    if not REPORTLAB_OK:
        return jsonify({'error': 'reportlab not installed'}), 500
    profile = request.json or session.get('profile', {})
    if not profile:
        return jsonify({'error': 'No profile data'}), 400
    try:
        buf  = generate_nutrition_pdf(profile)
        name = profile.get('name','user').replace(' ','_')
        return send_file(buf, mimetype='application/pdf', as_attachment=True,
            download_name=f'FitAI_Plan_{name}.pdf')
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
