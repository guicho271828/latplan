
-- find the configuration that solved the most


with
  best as (
    select heuristics, generator, comment, aeclass, time_start, found, max(valid), optimal, exhausted
      from (
        select heuristics,
               "parameters.generator"  as generator,
               "parameters.comment"    as comment,
               "parameters.aeclass"    as aeclass,
               "parameters.time_start" as time_start,
               sum(exhausted = "true") as exhausted,
               sum(found = "true") as found,
               sum(valid = "true") as valid,
               sum(valid = "true" and "statistics.plan_length" = cast(substr(problem, 1, 3) as int)) as optimal
          from planning
         where aeclass in (select distinct aeclass from AECLASS)
           and noise = NOISE
           and cycle = CYCLE
         group by heuristics, generator, comment, aeclass, time_start)
     group by heuristics, generator, comment, aeclass),
  statistics as (
    select heuristics,
           "parameters.generator"  as generator,
           "parameters.comment"    as comment,
           "parameters.aeclass"    as aeclass,
           "parameters.time_start" as time_start,
           domain, problem,
           ((valid = "true")*"statistics.generated")+((valid = "false")*(1000000000-random()%400000000))  as generated,
           ((valid = "true")*"statistics.expanded")+((valid = "false")*(1000000000-random()%400000000))  as expanded,
           ((valid = "true")*"statistics.evaluated")+((valid = "false")*(1000000000-random()%400000000))  as evaluated,
           ((valid = "true")*"statistics.search")+((valid = "false")*(10000-random()%2000))  as search
      from planning
     where noise = NOISE
       and cycle = CYCLE),
  best_stat as (
    select *
      from best join statistics
     where best.heuristics = statistics.heuristics
       and best.generator = statistics.generator
       and best.comment = statistics.comment
       and best.aeclass = statistics.aeclass
       and best.time_start = statistics.time_start),
  x as (
    select *
      from best_stat
     where comment = "kltune2"),
  y as (
    select *
      from best_stat
     where comment = "nozsae2")
select GROUP.id,
       x.METRIC, 0, y.METRIC, 0,
       x.problem, x.domain
  from x
         join y
             on  x.domain = y.domain
             and x.problem = y.problem
             and x.heuristics = y.heuristics
         join domains
             on x.domain = domains.key
         join generators
             on x.generator = generators.key
         join heuristics
             on x.heuristics = heuristics.key

