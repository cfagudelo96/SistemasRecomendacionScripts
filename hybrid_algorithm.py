from surprise import AlgoBase
import numpy as np
import _mysql


class HybridAlgorithm(AlgoBase):
    db = _mysql.connect(host="localhost", user="SistemasRecomendacionG1",
                        passwd="SistemasRecomendacionG1", db="yelp_db")

    def __init__(self, collaborative_weight=0):
        AlgoBase.__init__(self)
        self.collaborative_weight = collaborative_weight
        self.content_weight = 1 - collaborative_weight
        self.mean = 3

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
        self.mean = np.mean([r for (_, _, r) in self.trainset.all_ratings()])
        return self

    def estimate(self, uid, iid):
        try:
            u = self.trainset.to_raw_uid(uid)
            i = self.trainset.to_raw_iid(iid)

            collaborative_query = ("SELECT value FROM collaborative_recommendations r WHERE " +
                                   "r.user_id = \"" + u + "\"" +
                                   " AND r.business_id = \"" + i + "\"")
            HybridAlgorithm.db.query(collaborative_query)
            collaborative_result = self.db.store_result()
            collaborative_row = collaborative_result.fetch_row(how=1, maxrows=1)

            content_query = ("SELECT value FROM text_recommendations r WHERE " +
                             "r.user_id = \"" + u + "\"" +
                             " AND r.business_id = \"" + i + "\"" +
                             " AND r.category = 1")
            HybridAlgorithm.db.query(content_query)
            content_result = self.db.store_result()
            content_row = content_result.fetch_row(how=1, maxrows=1)

            if len(collaborative_row) == 0 and len(content_row) == 0:
                return self.mean
            elif len(collaborative_row) > 0 and len(content_row) == 0:
                return float(collaborative_row[0]["value"])
            elif len(collaborative_row) == 0 and len(content_row) > 0:
                return float(content_row[0]["value"])
            else:
                return float(collaborative_row[0]["value"]) * self.collaborative_weight + \
                       float(content_row[0]["value"]) * self.content_weight
        except Exception as e:
            return self.mean
