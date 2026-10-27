-- Query 1
SELECT CASE WHEN at_risk=1 THEN 'At Risk' ELSE 'Not At Risk' END AS grp,
       COUNT(*) AS n,
       ROUND(AVG(stress_level),2)        AS stress,
       ROUND(AVG(anxiety_score),2)       AS anxiety,
       ROUND(AVG(depression_score),2)    AS depression,
       ROUND(AVG(work_hours_per_week),1) AS work_hours,
       ROUND(AVG(overtime_hours),2)      AS overtime,
       ROUND(AVG(work_life_balance),2)   AS wlb,
       ROUND(AVG(manager_support),2)     AS mgr_support,
       ROUND(AVG(job_satisfaction),2)    AS job_sat,
       ROUND(AVG(sleep_hours),2)         AS sleep,
       ROUND(AVG(screen_time_hours),2)   AS screen,
       ROUND(AVG(social_support_score),2) AS social
FROM burnout_v
GROUP BY at_risk
ORDER BY at_risk DESC;

-- Query 2
SELECT feature, group_name,
       ROUND(r,3) AS corr_with_at_risk, ROUND(ABS(r),3) AS abs_corr
FROM (
  SELECT 'stress_level' feature,'Psychological' group_name, CORR(stress_level,at_risk) r FROM burnout_v UNION ALL
  SELECT 'anxiety_score','Psychological', CORR(anxiety_score,at_risk) FROM burnout_v UNION ALL
  SELECT 'depression_score','Psychological', CORR(depression_score,at_risk) FROM burnout_v UNION ALL
  SELECT 'work_hours_per_week','Work environment', CORR(work_hours_per_week,at_risk) FROM burnout_v UNION ALL
  SELECT 'overtime_hours','Work environment', CORR(overtime_hours,at_risk) FROM burnout_v UNION ALL
  SELECT 'meetings_per_day','Work environment', CORR(meetings_per_day,at_risk) FROM burnout_v UNION ALL
  SELECT 'deadlines_missed','Work environment', CORR(deadlines_missed,at_risk) FROM burnout_v UNION ALL
  SELECT 'job_satisfaction','Work environment', CORR(job_satisfaction,at_risk) FROM burnout_v UNION ALL
  SELECT 'manager_support','Work environment', CORR(manager_support,at_risk) FROM burnout_v UNION ALL
  SELECT 'work_life_balance','Work environment', CORR(work_life_balance,at_risk) FROM burnout_v UNION ALL
  SELECT 'sleep_hours','Lifestyle', CORR(sleep_hours,at_risk) FROM burnout_v UNION ALL
  SELECT 'physical_activity_days','Lifestyle', CORR(physical_activity_days,at_risk) FROM burnout_v UNION ALL
  SELECT 'screen_time_hours','Lifestyle', CORR(screen_time_hours,at_risk) FROM burnout_v UNION ALL
  SELECT 'caffeine_intake','Lifestyle', CORR(caffeine_intake,at_risk) FROM burnout_v UNION ALL
  SELECT 'social_support_score','Work environment', CORR(social_support_score,at_risk) FROM burnout_v
)
ORDER BY abs_corr DESC;

-- Query 3
SELECT CASE WHEN stress_level < 4 THEN '1. Stress < 4'
            WHEN stress_level < 6 THEN '2. Stress 4-6'
            WHEN stress_level < 8 THEN '3. Stress 6-8'
            ELSE '4. Stress >= 8' END AS stress_band,
       CASE WHEN work_hours_per_week <= 40 THEN 'a. <=40h'
            WHEN work_hours_per_week <= 50 THEN 'b. 41-50h'
            WHEN work_hours_per_week <= 60 THEN 'c. 51-60h'
            ELSE 'd. >60h' END AS hours_band,
       COUNT(*) AS n,
       ROUND(100.0*AVG(at_risk),1) AS pct_at_risk
FROM burnout_v
GROUP BY 1,2
HAVING COUNT(*) >= 30
ORDER BY 1,2;