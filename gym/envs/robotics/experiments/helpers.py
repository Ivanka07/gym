#!/usr/bin/env python
import math


def calc_goals(goals):
    for i in range(1, len(goals)):
        g1 = goals[i-1]
        g2 = goals[i]
        #x = math.fabs(g1[0] - g2[0])/3.0 
        y = math.fabs(g1[1] - g2[1])/3.0 
        z = math.fabs(g1[2] - g2[2])/3.0

        g3 = [g1[0], g1[1]+y, g1[2]+z]
        g4 = [g2[0], g2[1]-y, g2[2]-z]
        print(g3)
        print(g4)

goals = [[0.5981303969388876, 0.29991985778555486, 0.1652725415991557], [0.7152361864217875, 0.003399714038554852, 0.7670338779291559], [0.45236837062878754, -0.28031426747044513, 0.1652725415991557]]

calc_goals(goals)