-- ============================================================
-- TrendRadar - готовые запросы
-- Вставляй в Neon: SQL Editor -> New query -> Run
-- ============================================================

-- 1. Топ-20 игр по онлайну (последний снимок)
select g.name,
       s.playing        as ccu,
       s.visits,
       g.niche
from snapshots s
join games g on g.universe_id = s.universe_id
where s.ts = (select max(ts) from snapshots)
order by s.playing desc
limit 20;


-- 2. Кто вырос сильнее всего (первый снимок -> последний)
--    Даёт реальную картину, когда накопится история за день.
with bounds as (
    select min(ts) as t_first, max(ts) as t_last from snapshots
)
select g.name,
       f.playing as ccu_before,
       l.playing as ccu_now,
       round(100.0 * (l.playing - f.playing) / nullif(f.playing, 0), 1) as growth_pct
from snapshots l
join snapshots f on f.universe_id = l.universe_id
join games g      on g.universe_id = l.universe_id
cross join bounds b
where l.ts = b.t_last
  and f.ts = b.t_first
  and f.playing > 1000              -- мелочь не интересна
order by growth_pct desc
limit 20;


-- 3. Распределение по нишам
select coalesce(g.niche, '?') as niche,
       count(*)               as games,
       sum(s.playing)         as total_ccu
from snapshots s
join games g on g.universe_id = s.universe_id
where s.ts = (select max(ts) from snapshots)
group by 1
order by total_ccu desc;


-- 4. Игры, которые упоминают на YouTube
select g.name,
       count(*)          as videos,
       max(m.method)     as how_matched
from mentions m
join games g on g.universe_id = m.universe_id
group by g.name
order by videos desc
limit 20;


-- 5. Самые быстрые видео прямо сейчас
select v.title,
       v.views,
       round(v.views_per_hour) as views_per_hour,
       v.published
from yt_videos v
order by v.views_per_hour desc
limit 20;


-- 6. Внешние сигналы по источникам
select source,
       count(*)        as signals,
       max(observed)   as last_update
from origin_signal
group by source
order by signals desc;


-- 7. Сколько всего накопилось
select 'games'         as table_name, count(*) as rows from games
union all select 'snapshots',     count(*) from snapshots
union all select 'yt_channels',   count(*) from yt_channels
union all select 'yt_videos',     count(*) from yt_videos
union all select 'mentions',      count(*) from mentions
union all select 'origin_signal', count(*) from origin_signal
order by rows desc;


-- 8. История одной игры (подставь своё название)
select g.name,
       s.ts,
       s.playing as ccu,
       s.visits
from snapshots s
join games g on g.universe_id = s.universe_id
where g.name ilike '%egg%'         -- <-- тут искомое слово
order by s.ts desc
limit 100;


-- 9. Прирост за 15 минут (для свежих данных)
--    Требует историю: сработает после нескольких запусков workflow.
with ranked as (
    select universe_id, ts, playing,
           lag(playing) over (partition by universe_id order by ts) as prev
    from snapshots
    where ts > now() - interval '2 hours'
)
select g.name,
       r.prev as was,
       r.playing as now,
       r.playing - r.prev as delta
from ranked r
join games g on g.universe_id = r.universe_id
where r.prev is not null
  and r.prev > 1000
order by delta desc
limit 20;
