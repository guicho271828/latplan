
-- find the configuration that solved the most


with
  best as (
    select domain, heuristics,
           "parameters.generator"  as generator,
           "parameters.comment"    as comment,
           "parameters.aeclass"    as aeclass,
           "parameters.beta_d" as beta_d,
           "parameters.beta_z" as beta_z,
           "parameters.N" as N,
           "parameters.time_start" as time_start,
           sum(exhausted = "true") as exhausted,
           sum(found = "true") as found,
           sum(valid = "true") as valid,
           sum(valid = "true" and "statistics.plan_length" = cast(substr(problem, 1, 3) as int)) as optimal
      from planning
     where aeclass in (select distinct aeclass from AECLASS)
       and cycle = CYCLE
       and noise = NOISE
       and heuristics in (select key from heuristics3)
     group by domain, heuristics, generator, comment, aeclass, time_start),
  x as (
    select *
      from best
     where comment = "kltune2"),
  y as (
    select *
      from best
     where comment = "nozsae2")
select GROUP.id,
       x.METRIC, 0, y.METRIC, 0
  from x
         join y
             on  x.domain = y.domain
             and x.heuristics = y.heuristics
             and x.N = y.N
             and x.beta_d = y.beta_d
             and x.beta_z = y.beta_z
         join domains
             on x.domain = domains.key
         join generators
             on x.generator = generators.key
         join heuristics3
             on x.heuristics = heuristics3.key


