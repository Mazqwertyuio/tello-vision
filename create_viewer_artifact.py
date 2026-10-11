"""Generate self-contained HTML viewer artifact with embedded navigation frames (including floor exclusion)."""

import base64
from pathlib import Path

art_dir = Path(r"C:\Users\User\.gemini\antigravity\brain\28716e03-4c98-47b6-a038-f3f82a5ddf33")

wall_b64 = base64.b64encode(open(art_dir / "01_flat_wall_brake.jpg", "rb").read()).decode("utf-8")
aisle_b64 = base64.b64encode(open(art_dir / "02_clear_aisle_forward.jpg", "rb").read()).decode("utf-8")
centering_b64 = base64.b64encode(open(art_dir / "03_corridor_centering.jpg", "rb").read()).decode("utf-8")
door_b64 = base64.b64encode(open(art_dir / "04_closed_door_brake.jpg", "rb").read()).decode("utf-8")
f180_b64 = base64.b64encode(open(art_dir / "05_frame_180_flight_window.jpg", "rb").read()).decode("utf-8")

html = f"""<!DOCTYPE html>
<html lang="uk">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Tello Vision DA2 Navigation Viewer</title>
  <script src="https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js"></script>
  <style>
    :root {{
      --bg: #0f172a;
      --card: #1e293b;
      --border: #334155;
      --text: #f8fafc;
      --muted: #94a3b8;
      --accent: #38bdf8;
    }}
    body {{
      background-color: var(--bg);
      color: var(--text);
      font-family: ui-sans-serif, system-ui, -apple-system, sans-serif;
    }}
  </style>
</head>
<body class="p-3 max-w-5xl mx-auto">
  <div class="bg-slate-800/80 border border-slate-700 rounded-xl p-4 shadow-2xl mb-4 backdrop-blur">
    <div class="flex flex-wrap items-center justify-between gap-3 mb-4">
      <div>
        <h1 class="text-lg font-bold text-sky-400 flex items-center gap-2">
          <span>🚁</span> Tello Vision: Візуалізація навігації DA2 (Відсікання підлоги)
        </h1>
        <p class="text-xs text-slate-400 mt-1">
          Відео: <code class="bg-slate-900 px-2 py-0.5 rounded text-amber-300 font-mono">debug/20261009_154542</code> | Зона оцінки: <span class="text-emerald-400 font-semibold">Тільки центральне вікно на рівні очей (18% - 54%)</span>
        </p>
      </div>
      <div class="flex items-center gap-2">
        <span class="text-xs px-2.5 py-1 rounded-full font-semibold bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
          Depth Anything V2 (CUDA)
        </span>
        <span class="text-xs px-2.5 py-1 rounded-full font-semibold bg-sky-500/20 text-sky-400 border border-sky-500/30">
          ~17.5 ms (55+ FPS)
        </span>
      </div>
    </div>

    <!-- Tab navigation -->
    <div class="flex flex-wrap gap-2 mb-3 border-b border-slate-700/60 pb-3">
      <button onclick="selectStage(0)" id="tab-0" class="stage-btn px-3 py-1.5 rounded-lg text-xs font-medium bg-sky-500 text-white transition">
        1. Кадр 180 (Підлога відсічена - Вільний політ!)
      </button>
      <button onclick="selectStage(1)" id="tab-1" class="stage-btn px-3 py-1.5 rounded-lg text-xs font-medium bg-slate-700/60 text-slate-300 hover:bg-slate-700 transition">
        2. Однотонна стіна (FLAT_WALL_BRAKE)
      </button>
      <button onclick="selectStage(2)" id="tab-2" class="stage-btn px-3 py-1.5 rounded-lg text-xs font-medium bg-slate-700/60 text-slate-300 hover:bg-slate-700 transition">
        3. Вільний коридор (CLEAR_AISLE)
      </button>
      <button onclick="selectStage(3)" id="tab-3" class="stage-btn px-3 py-1.5 rounded-lg text-xs font-medium bg-slate-700/60 text-slate-300 hover:bg-slate-700 transition">
        4. Центрування (CORRIDOR CENTERING)
      </button>
      <button onclick="selectStage(4)" id="tab-4" class="stage-btn px-3 py-1.5 rounded-lg text-xs font-medium bg-slate-700/60 text-slate-300 hover:bg-slate-700 transition">
        5. Зачинені двері (CLOSED_DOOR)
      </button>
    </div>

    <!-- Active Stage Viewer -->
    <div class="bg-slate-900/90 rounded-lg p-3 border border-slate-700">
      <div class="relative overflow-hidden rounded border border-slate-800 shadow-lg">
        <img id="stage-img" src="data:image/jpeg;base64,{f180_b64}" alt="Navigation Frame" class="w-full h-auto object-contain">
      </div>
      <div class="mt-3 p-3 bg-slate-800/60 rounded border border-slate-700/50">
        <div class="flex items-center justify-between mb-2">
          <span id="stage-badge" class="text-xs px-2.5 py-0.5 rounded font-bold bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
            CLEAR_AISLE [ВІКНО ПОЛЬОТУ ВІДКРИТЕ]
          </span>
          <span id="stage-timing" class="text-xs font-mono text-slate-400">Кадр: 180 / 330</span>
        </div>
        <p id="stage-desc" class="text-xs sm:text-sm text-slate-200 leading-relaxed"></p>
        <div class="mt-3 grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs font-mono">
          <div class="bg-slate-900 p-2 rounded border border-slate-800">
            <div class="text-slate-400 text-[11px]">Команда Forward</div>
            <div id="cmd-fb" class="text-emerald-400 font-bold text-xs sm:text-sm">Fwd: +18 (Повний хід)</div>
          </div>
          <div class="bg-slate-900 p-2 rounded border border-slate-800">
            <div class="text-slate-400 text-[11px]">Команда Strafe/Yaw</div>
            <div id="cmd-yaw" class="text-sky-400 font-bold text-xs sm:text-sm">Yaw: +20, Strafe: +18</div>
          </div>
          <div class="bg-slate-900 p-2 rounded border border-slate-800">
            <div class="text-slate-400 text-[11px]">Сектор польоту (Floor Excluded)</div>
            <div id="sec-val" class="text-emerald-400 font-bold text-xs sm:text-sm">C=28 (Чистий коридор)</div>
          </div>
          <div class="bg-slate-900 p-2 rounded border border-slate-800">
            <div class="text-slate-400 text-[11px]">Зона оцінки</div>
            <div id="door-val" class="text-amber-400 font-bold text-xs sm:text-sm">Eye-Level (18%-54%)</div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <script>
    const stages = [
      {{
        badge: 'CLEAR_AISLE [ВІКНО ПОЛЬОТУ ВІДКРИТЕ]',
        badgeColor: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30',
        timing: 'Кадр: 180 / 330 (Ваш скріншот)',
        desc: 'ПОВНЕ ВІДСІКАННЯ ПІДЛОГИ: Зелена рамка тепер аналізує ТІЛЬКИ центральне вікно польоту на рівні очей дрона (18% - 54% по висоті). Жовта підлога внизу затінена та ігнорується ([FLOOR EXCLUDED]). Оскільки вікно польоту абсолютно темне (чистий простір), автопілот не гальмує і летить уперед на повній швидкості (Fwd: +18)!',
        fb: 'Fwd: +18 (Вільний політ)',
        yaw: 'Yaw: +20 | Strafe: +18',
        sec: 'L:120 | C:28 | R:45 (C_clean=28)',
        door: 'Підлога відсічена',
        img: 'data:image/jpeg;base64,{f180_b64}'
      }},
      {{
        badge: 'FLAT_WALL_BRAKE',
        badgeColor: 'bg-red-500/20 text-red-400 border-red-500/30',
        timing: 'Кадр: 5 / 200 | Час: 0.17s',
        desc: 'Безтекстурна плоска стіна прямо перед камерою на рівні очей. Wall Guard виявив нульову текстуру та небезпеку в польотному вікні -> екстрене гальмування (Fwd: 0) та розворот.',
        fb: 'Fwd: 0 (Блокування)',
        yaw: 'Yaw: +24 | Strafe: +12',
        sec: 'L:94 | C:94 | R:92',
        door: 'NO_DOOR',
        img: 'data:image/jpeg;base64,{wall_b64}'
      }},
      {{
        badge: 'CLEAR_AISLE',
        badgeColor: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30',
        timing: 'Кадр: 45 / 200 | Час: 1.50s',
        desc: 'Вільний коридор на висоті 30 см. Вікно польоту повністю темне (прохід углиб). Зелений AR-коридор плавно веде дрон уперед крізь проходи.',
        fb: 'Fwd: +18 (Повний хід)',
        yaw: 'Yaw: 0 | Strafe: 0',
        sec: 'L:78 | C:58 | R:92',
        door: 'NO_DOOR',
        img: 'data:image/jpeg;base64,{aisle_b64}'
      }},
      {{
        badge: 'CORRIDOR CENTERING',
        badgeColor: 'bg-amber-500/20 text-amber-400 border-amber-500/30',
        timing: 'Кадр: 75 / 200 | Час: 2.50s',
        desc: 'Звуження коридору: лівий сектор наближається у польотному вікні. Дрон формує плавне підрулювання праворуч: Strafe: +14, Yaw: +16, безпечно огинаючи перешкоди.',
        fb: 'Fwd: +18 (Коридор)',
        yaw: 'Yaw: +16 | Strafe: +14',
        sec: 'L:115 | C:64 | R:70',
        door: 'NO_DOOR',
        img: 'data:image/jpeg;base64,{centering_b64}'
      }},
      {{
        badge: 'CLOSED_DOOR',
        badgeColor: 'bg-purple-500/20 text-purple-400 border-purple-500/30',
        timing: 'Кадр: 105 / 200 | Час: 3.50s',
        desc: 'Детекція дверей: контур дверей перевірено за вікном польоту DA2. Полотно дверей монолітне зі стіною -> вхід зачинено (Fwd: 0, утримання позиції).',
        fb: 'Fwd: 0 (Зачинені двері)',
        yaw: 'Yaw: 0 | Strafe: 0',
        sec: 'L:128 | C:76 | R:71',
        door: 'CLOSED_DOOR',
        img: 'data:image/jpeg;base64,{door_b64}'
      }}
    ];

    function selectStage(idx) {{
      document.querySelectorAll('.stage-btn').forEach((btn, i) => {{
        if (i === idx) {{
          btn.className = 'stage-btn px-3 py-1.5 rounded-lg text-xs font-medium bg-sky-500 text-white transition shadow';
        }} else {{
          btn.className = 'stage-btn px-3 py-1.5 rounded-lg text-xs font-medium bg-slate-700/60 text-slate-300 hover:bg-slate-700 transition';
        }}
      }});

      const s = stages[idx];
      document.getElementById('stage-img').src = s.img;
      document.getElementById('stage-badge').innerText = s.badge;
      document.getElementById('stage-badge').className = 'text-xs px-2.5 py-0.5 rounded font-bold border ' + s.badgeColor;
      document.getElementById('stage-timing').innerText = s.timing;
      document.getElementById('stage-desc').innerText = s.desc;
      document.getElementById('cmd-fb').innerText = s.fb;
      document.getElementById('cmd-yaw').innerText = s.yaw;
      document.getElementById('sec-val').innerText = s.sec;
      document.getElementById('door-val').innerText = s.door;
    }}

    selectStage(0);
  </script>
</body>
</html>"""

out_path = art_dir / "da2_navigation_viewer.html"
with open(out_path, "w", encoding="utf-8") as f:
    f.write(html)
print(f"Generated {out_path} ({len(html)} bytes)")
