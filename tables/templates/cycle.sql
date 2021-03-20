
-- This is ugly but I was lazy about making a better sql generator

-- parameter : WHAT \in xs01 ... zs34, AECLASS, GENERATOR, COMMENT

with
  xs01 as (
    select generator, time_start, comment,  0 as k, ("cycle_consistency.test.xs01.0") as v
      from AECLASS
     union
    select generator, time_start, comment,  1 as k, ("cycle_consistency.test.xs01.1") as v
      from AECLASS
     union
    select generator, time_start, comment,  2 as k, ("cycle_consistency.test.xs01.2") as v
      from AECLASS
     union
    select generator, time_start, comment,  3 as k, ("cycle_consistency.test.xs01.3") as v
      from AECLASS
     union
    select generator, time_start, comment,  4 as k, ("cycle_consistency.test.xs01.4") as v
      from AECLASS
     union
    select generator, time_start, comment,  5 as k, ("cycle_consistency.test.xs01.5") as v
      from AECLASS
     union
    select generator, time_start, comment,  6 as k, ("cycle_consistency.test.xs01.6") as v
      from AECLASS
     union
    select generator, time_start, comment,  7 as k, ("cycle_consistency.test.xs01.7") as v
      from AECLASS
     union
    select generator, time_start, comment,  8 as k, ("cycle_consistency.test.xs01.8") as v
      from AECLASS
     union
    select generator, time_start, comment,  9 as k, ("cycle_consistency.test.xs01.9") as v
      from AECLASS
  ),
  xs12 as (
    select generator, time_start, comment,  0 as k, ("cycle_consistency.test.xs12.0") as v
      from AECLASS
     union
    select generator, time_start, comment,  1 as k, ("cycle_consistency.test.xs12.1") as v
      from AECLASS
     union
    select generator, time_start, comment,  2 as k, ("cycle_consistency.test.xs12.2") as v
      from AECLASS
     union
    select generator, time_start, comment,  3 as k, ("cycle_consistency.test.xs12.3") as v
      from AECLASS
     union
    select generator, time_start, comment,  4 as k, ("cycle_consistency.test.xs12.4") as v
      from AECLASS
     union
    select generator, time_start, comment,  5 as k, ("cycle_consistency.test.xs12.5") as v
      from AECLASS
     union
    select generator, time_start, comment,  6 as k, ("cycle_consistency.test.xs12.6") as v
      from AECLASS
     union
    select generator, time_start, comment,  7 as k, ("cycle_consistency.test.xs12.7") as v
      from AECLASS
     union
    select generator, time_start, comment,  8 as k, ("cycle_consistency.test.xs12.8") as v
      from AECLASS
     union
    select generator, time_start, comment,  9 as k, ("cycle_consistency.test.xs12.9") as v
      from AECLASS
  ),
  xs13 as (
    select generator, time_start, comment,  0 as k, ("cycle_consistency.test.xs13.0") as v
      from AECLASS
     union
    select generator, time_start, comment,  1 as k, ("cycle_consistency.test.xs13.1") as v
      from AECLASS
     union
    select generator, time_start, comment,  2 as k, ("cycle_consistency.test.xs13.2") as v
      from AECLASS
     union
    select generator, time_start, comment,  3 as k, ("cycle_consistency.test.xs13.3") as v
      from AECLASS
     union
    select generator, time_start, comment,  4 as k, ("cycle_consistency.test.xs13.4") as v
      from AECLASS
     union
    select generator, time_start, comment,  5 as k, ("cycle_consistency.test.xs13.5") as v
      from AECLASS
     union
    select generator, time_start, comment,  6 as k, ("cycle_consistency.test.xs13.6") as v
      from AECLASS
     union
    select generator, time_start, comment,  7 as k, ("cycle_consistency.test.xs13.7") as v
      from AECLASS
     union
    select generator, time_start, comment,  8 as k, ("cycle_consistency.test.xs13.8") as v
      from AECLASS
     union
    select generator, time_start, comment,  9 as k, ("cycle_consistency.test.xs13.9") as v
      from AECLASS
  ),
  xs34 as (
    select generator, time_start, comment,  0 as k, ("cycle_consistency.test.xs34.0") as v
      from AECLASS
     union
    select generator, time_start, comment,  1 as k, ("cycle_consistency.test.xs34.1") as v
      from AECLASS
     union
    select generator, time_start, comment,  2 as k, ("cycle_consistency.test.xs34.2") as v
      from AECLASS
     union
    select generator, time_start, comment,  3 as k, ("cycle_consistency.test.xs34.3") as v
      from AECLASS
     union
    select generator, time_start, comment,  4 as k, ("cycle_consistency.test.xs34.4") as v
      from AECLASS
     union
    select generator, time_start, comment,  5 as k, ("cycle_consistency.test.xs34.5") as v
      from AECLASS
     union
    select generator, time_start, comment,  6 as k, ("cycle_consistency.test.xs34.6") as v
      from AECLASS
     union
    select generator, time_start, comment,  7 as k, ("cycle_consistency.test.xs34.7") as v
      from AECLASS
     union
    select generator, time_start, comment,  8 as k, ("cycle_consistency.test.xs34.8") as v
      from AECLASS
     union
    select generator, time_start, comment,  9 as k, ("cycle_consistency.test.xs34.9") as v
      from AECLASS
  ),
  zs01 as (
    select generator, time_start, comment,  0 as k, ("cycle_consistency.test.zs01.0") as v
      from AECLASS
     union
    select generator, time_start, comment,  1 as k, ("cycle_consistency.test.zs01.1") as v
      from AECLASS
     union
    select generator, time_start, comment,  2 as k, ("cycle_consistency.test.zs01.2") as v
      from AECLASS
     union
    select generator, time_start, comment,  3 as k, ("cycle_consistency.test.zs01.3") as v
      from AECLASS
     union
    select generator, time_start, comment,  4 as k, ("cycle_consistency.test.zs01.4") as v
      from AECLASS
     union
    select generator, time_start, comment,  5 as k, ("cycle_consistency.test.zs01.5") as v
      from AECLASS
     union
    select generator, time_start, comment,  6 as k, ("cycle_consistency.test.zs01.6") as v
      from AECLASS
     union
    select generator, time_start, comment,  7 as k, ("cycle_consistency.test.zs01.7") as v
      from AECLASS
     union
    select generator, time_start, comment,  8 as k, ("cycle_consistency.test.zs01.8") as v
      from AECLASS
     union
    select generator, time_start, comment,  9 as k, ("cycle_consistency.test.zs01.9") as v
      from AECLASS
  ),
  zs12 as (
    select generator, time_start, comment,  0 as k, ("cycle_consistency.test.zs12.0") as v
      from AECLASS
     union
    select generator, time_start, comment,  1 as k, ("cycle_consistency.test.zs12.1") as v
      from AECLASS
     union
    select generator, time_start, comment,  2 as k, ("cycle_consistency.test.zs12.2") as v
      from AECLASS
     union
    select generator, time_start, comment,  3 as k, ("cycle_consistency.test.zs12.3") as v
      from AECLASS
     union
    select generator, time_start, comment,  4 as k, ("cycle_consistency.test.zs12.4") as v
      from AECLASS
     union
    select generator, time_start, comment,  5 as k, ("cycle_consistency.test.zs12.5") as v
      from AECLASS
     union
    select generator, time_start, comment,  6 as k, ("cycle_consistency.test.zs12.6") as v
      from AECLASS
     union
    select generator, time_start, comment,  7 as k, ("cycle_consistency.test.zs12.7") as v
      from AECLASS
     union
    select generator, time_start, comment,  8 as k, ("cycle_consistency.test.zs12.8") as v
      from AECLASS
     union
    select generator, time_start, comment,  9 as k, ("cycle_consistency.test.zs12.9") as v
      from AECLASS
  ),
  zs13 as (
    select generator, time_start, comment,  0 as k, ("cycle_consistency.test.zs13.0") as v
      from AECLASS
     union
    select generator, time_start, comment,  1 as k, ("cycle_consistency.test.zs13.1") as v
      from AECLASS
     union
    select generator, time_start, comment,  2 as k, ("cycle_consistency.test.zs13.2") as v
      from AECLASS
     union
    select generator, time_start, comment,  3 as k, ("cycle_consistency.test.zs13.3") as v
      from AECLASS
     union
    select generator, time_start, comment,  4 as k, ("cycle_consistency.test.zs13.4") as v
      from AECLASS
     union
    select generator, time_start, comment,  5 as k, ("cycle_consistency.test.zs13.5") as v
      from AECLASS
     union
    select generator, time_start, comment,  6 as k, ("cycle_consistency.test.zs13.6") as v
      from AECLASS
     union
    select generator, time_start, comment,  7 as k, ("cycle_consistency.test.zs13.7") as v
      from AECLASS
     union
    select generator, time_start, comment,  8 as k, ("cycle_consistency.test.zs13.8") as v
      from AECLASS
     union
    select generator, time_start, comment,  9 as k, ("cycle_consistency.test.zs13.9") as v
      from AECLASS
  ),
  zs34 as (
    select generator, time_start, comment,  0 as k, ("cycle_consistency.test.zs34.0") as v
      from AECLASS
     union
    select generator, time_start, comment,  1 as k, ("cycle_consistency.test.zs34.1") as v
      from AECLASS
     union
    select generator, time_start, comment,  2 as k, ("cycle_consistency.test.zs34.2") as v
      from AECLASS
     union
    select generator, time_start, comment,  3 as k, ("cycle_consistency.test.zs34.3") as v
      from AECLASS
     union
    select generator, time_start, comment,  4 as k, ("cycle_consistency.test.zs34.4") as v
      from AECLASS
     union
    select generator, time_start, comment,  5 as k, ("cycle_consistency.test.zs34.5") as v
      from AECLASS
     union
    select generator, time_start, comment,  6 as k, ("cycle_consistency.test.zs34.6") as v
      from AECLASS
     union
    select generator, time_start, comment,  7 as k, ("cycle_consistency.test.zs34.7") as v
      from AECLASS
     union
    select generator, time_start, comment,  8 as k, ("cycle_consistency.test.zs34.8") as v
      from AECLASS
     union
    select generator, time_start, comment,  9 as k, ("cycle_consistency.test.zs34.9") as v
      from AECLASS
  ),
  best as (
    select generator, comment, time_start, min("metrics.test.elbo")
      from AECLASS
     group by generator, comment),
  cycle as (
    select best.generator as generator,
           best.time_start as start_time,
           best.comment as comment,
           xs01.k as k,
           max(1e-14,xs01.v) as xs01,
           max(1e-14,xs12.v) as xs12,
           max(1e-14,xs13.v) as xs13,
           max(1e-14,xs34.v) as xs34,
           max(1e-14,zs01.v) as zs01,
           max(1e-14,zs12.v) as zs12,
           max(1e-14,zs13.v) as zs13,
           max(1e-14,zs34.v) as zs34
      from best
             join xs01 on best.generator = xs01.generator and best.time_start = xs01.time_start and best.comment = xs01.comment
             join xs12 on best.generator = xs12.generator and best.time_start = xs12.time_start and best.comment = xs12.comment and xs01.k = xs12.k
             join xs13 on best.generator = xs13.generator and best.time_start = xs13.time_start and best.comment = xs13.comment and xs01.k = xs13.k
             join xs34 on best.generator = xs34.generator and best.time_start = xs34.time_start and best.comment = xs34.comment and xs01.k = xs34.k
             join zs01 on best.generator = zs01.generator and best.time_start = zs01.time_start and best.comment = zs01.comment and xs01.k = zs01.k
             join zs12 on best.generator = zs12.generator and best.time_start = zs12.time_start and best.comment = zs12.comment and xs01.k = zs12.k
             join zs13 on best.generator = zs13.generator and best.time_start = zs13.time_start and best.comment = zs13.comment and xs01.k = zs13.k
             join zs34 on best.generator = zs34.generator and best.time_start = zs34.time_start and best.comment = zs34.comment and xs01.k = zs34.k)
-- select generator, time_start, comment,  0 as k, ("cycle_consistency.test.xs01.0") as v
--   from AECLASS
-- select *
--   from xs01
-- select *
--   from cycle
select k, xs01, xs12, xs13, xs34, zs01, zs12, zs13, zs34
  from cycle
 where comment = COMMENT
   and generator = GENERATOR
   order by k
