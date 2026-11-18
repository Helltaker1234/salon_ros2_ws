import ast,collections,pathlib,sys,time,types,unittest
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from physics_atv_visual_mapping.processing_schedule import CloudSchedule
SOURCE=ROOT/'scripts/ros/dino_localmapping.py'
# Exercise the deployed input callbacks without importing ROS/GPU/model weights.
tree=ast.parse(SOURCE.read_text());target=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='DinoMappingNode')
names={'_stamp_to_ns','_stamp_to_sec','_pcl_callback','_odom_callback','_image_callback','_request_processing','_nearest_message','_process_latest_inputs'}
cls=ast.ClassDef(name='InputCallbacks',bases=[],keywords=[],body=[n for n in target.body if isinstance(n,ast.FunctionDef) and n.name in names],decorator_list=[])
module=ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[]));namespace={'time':time};exec(compile(module,str(SOURCE),'exec'),namespace);Callbacks=namespace['InputCallbacks']
def msg(ns):
 return types.SimpleNamespace(header=types.SimpleNamespace(stamp=types.SimpleNamespace(sec=ns//10**9,nanosec=ns%10**9),frame_id='camera_init'))
def node():
 n=Callbacks();n.latest_pcl=None;n.last_processed_pcl_stamp_ns=None;n.cloud_schedule=CloudSchedule(10);n.pending_since=None;n.processing=False;n.max_image_dt=.08;n.max_odom_dt=.12;n.sync_wait_s=0;n.pcl_frame_remap=None;n.odom_frame=None;n.image_cache=collections.deque(maxlen=10);n.odom_cache=collections.deque(maxlen=10);n.scheduling_stats=collections.Counter();n.processing_times_ms=collections.deque(maxlen=2048);n.pair_deltas_ms=collections.deque(maxlen=2048);n.triggers=0;n.published=[]
 def trigger():n.triggers+=1
 n.process_guard=types.SimpleNamespace(trigger=trigger)
 def infer():
  assert n.processing
  n._process_latest_inputs()  # Re-entry cannot start another inference.
  n.published.append(n._stamp_to_ns(n.pcl_msg.header.stamp))
 n.spin_once=infer;return n
class ScheduleTests(unittest.TestCase):
 def test_ten_hz_jitter_keeps_every_cloud(self):
  s=CloudSchedule(10)
  for i in range(1000):
   stamp=10**9+i*100_000_000+[0,-3,4,-2,0][i%5]*1_000_000
   self.assertTrue(s.ready(stamp));s.mark(stamp);self.assertFalse(s.ready(stamp))
 def test_higher_input_rate_is_bounded(self):
  s=CloudSchedule(10);count=0
  for stamp in range(0,20*10**9,10_000_000):
   if s.ready(stamp):s.mark(stamp);count+=1
  self.assertTrue(200<=count<=201,count)
 def test_five_hz_configuration_keeps_its_budget(self):
  s=CloudSchedule(5);count=0
  for stamp in range(0,60*10**9,100_000_000):
   if s.ready(stamp):s.mark(stamp);count+=1
  self.assertEqual(count,300)
 def test_late_cloud_skips_old_deadlines_without_burst(self):
  s=CloudSchedule(10);s.mark(10**9);s.mark(11*10**9)
  self.assertFalse(s.ready(11*10**9+1));self.assertFalse(s.ready(10**9))
 def test_callback_order_does_not_lose_a_period(self):
  n=node()
  for i in range(100):
   stamp=10**9+i*100_000_000;n._process_latest_inputs()
   n._pcl_callback(msg(stamp));n._process_latest_inputs()
   n._image_callback(msg(stamp));n._process_latest_inputs()
   n._odom_callback(msg(stamp));n._process_latest_inputs()
  self.assertEqual(len(n.published),100);self.assertEqual(len(set(n.published)),100)
 def test_waits_for_better_pair_and_keeps_error_bounds(self):
  n=node();n.sync_wait_s=.05;t=10**9
  n._odom_callback(msg(t));n._image_callback(msg(t-20_000_000));n._pcl_callback(msg(t));n._process_latest_inputs();self.assertFalse(n.published)
  n._image_callback(msg(t+10_000_000));n._process_latest_inputs();self.assertEqual(n.published,[t]);self.assertEqual(n.pair_deltas_ms[-1],(0.,10.))
 def test_sync_timeout_never_relaxes_timestamp_limits(self):
  n=node();n._odom_callback(msg(10**9));n._image_callback(msg(500_000_000));n._pcl_callback(msg(10**9));n.pending_since-=1;n._process_latest_inputs();self.assertFalse(n.published)
  n._image_callback(msg(10**9));n._process_latest_inputs();self.assertEqual(n.published,[10**9])
if __name__=='__main__':unittest.main()
