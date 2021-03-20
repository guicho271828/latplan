
-- find the configuration that solved the most


with
  best as (
    select domain, heuristics, cycle, noise, comment, aeclass, time_start, max(found) as found, valid, optimal, exhausted
      from (
        select domain, heuristics, cycle, noise,
               "parameters.comment"    as comment,
               "parameters.aeclass"    as aeclass,
               "parameters.time_start" as time_start,
               sum(exhausted = "true") as exhausted,
               sum(found = "true") as found,
               sum(valid = "true") as valid,
               sum(valid = "true" and "statistics.plan_length" = cast(substr(problem, 1, 3) as int)) as optimal
          from planning
         group by domain, heuristics, cycle, noise, comment, aeclass, time_start)
     group by domain, heuristics, cycle, noise, comment, aeclass)
select aeclass.name as aeclass,
       domains.name as domain,
       domains.id as domainid,
       comments.name as comment,
       heuristics.name as heuristics,
       cycle, noise, found, valid, optimal, valid-optimal as suboptimal, exhausted
  from best
         join domains
             on domains.key = best.domain
         join heuristics
             on heuristics.key = best.heuristics
         join comments
             on comments.key = best.comment
         join aeclass
             on aeclass.key = best.aeclass
         
