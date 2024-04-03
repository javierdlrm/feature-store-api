import hopsworks

project = hopsworks.login()
fs = project.get_feature_store()
mr = project.get_model_registry()
ms = project.get_model_serving()

m = ms.get_deployment("dehandmeventsredirectdeployment")
m.delete()
m = ms.get_deployment("dehandmquerydeployment")
m.delete()
m = ms.get_deployment("dehandmrankingdeployment")
m.delete()

m = mr.get_model("de_h_and_m_events_redirect", 1)
m.delete()
m = mr.get_model("de_h_and_m_ranking_model", 1)
m.delete()
m = mr.get_model("de_h_and_m_query_model", 1)
m.delete()
m = mr.get_model("de_h_and_m_candidate_model", 1)
m.delete()

fv = fs.get_feature_view("de_h_and_m_items", 1)
fv.delete()
fv = fs.get_feature_view("de_h_and_m_events", 1)
fv.delete()

fg = fs.get_feature_group("de_h_and_m_items", 1)
fg.delete()
fg = fs.get_feature_group("de_h_and_m_events", 1)
fg.delete()
fg = fs.get_feature_group("de_h_and_m_decisions", 1)
fg.delete()