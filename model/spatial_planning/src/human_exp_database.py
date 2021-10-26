
class HumanExperimentResults():
    def __init__(self):
        self.datas = {}

    def add_result(self, tmap, patient_id, states, real_states, actions, traj, hdict):
        if(tmap not in self.datas.keys()):
            self.datas[tmap] = {}

        if(patient_id not in self.datas[tmap].keys()):
            self.datas[tmap][patient_id] = {}

        self.datas[tmap][patient_id] = (states, real_states, actions, traj, hdict)