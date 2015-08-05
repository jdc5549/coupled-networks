load Polish_ps;

ps = updateps(ps);
ps = redispatch(ps);
ps = dcpf(ps);
ps.bus(:,C.bu.power_from_sh) = assign_loads_to_buses(ps);

save Polish_ps ps;
