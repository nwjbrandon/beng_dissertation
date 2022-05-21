import tkinter as tk

from pymycobot.mycobot import MyCobot

start_angles = [0, 0, 0, 0, 0, 0]
end_angles = [153.19, 137.81, -153.54, 156.79, 87.27, 13.62]


class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master

        self.joint_slider_start_pos = (0, 50)
        self.joint_control_values = [0] * 6
        self.pos_slider_start_pos = (0, 200)
        self.pos_control_values = [0] * 6

        self.create_widgets()

    def create_widgets(self):
        self.create_joint_control_widgets()
        self.create_pos_control_widgets()

        self.start_btn = tk.Button(self.master, text="Start Pos")
        self.start_btn["command"] = self.move_to_start_configuration
        self.start_btn.place(x=50, y=350)

        self.end_btn = tk.Button(self.master, text="End Pos")
        self.end_btn["command"] = self.move_to_end_configuration
        self.end_btn.place(x=150, y=350)

        self.stop_btn = tk.Button(self.master, text="Stop")
        self.stop_btn["command"] = self.stop_moving
        self.stop_btn.place(x=375, y=350)

        self.release_btn = tk.Button(self.master, text="Release")
        self.release_btn["command"] = self.release_servos
        self.release_btn.place(x=450, y=350)

    def move_to_start_configuration(self):
        print("move to start configuration")
        mycobot.send_angles(start_angles, 50)

    def move_to_end_configuration(self):
        print("move to end configuration")
        mycobot.send_angles(end_angles, 50)

    def stop_moving(self):
        print("stop moving")
        mycobot.stop()

    def release_servos(self):
        print("release servos")
        mycobot.release_all_servos()

    def create_joint_control_widgets(self):
        self.joint1 = tk.Scale(self.master, variable=tk.IntVar(), from_=-180, to=180)
        self.joint1["command"] = lambda x: self.set_joint_values(0, x)
        self.joint1.place(x=self.joint_slider_start_pos[0] + 50, y=self.joint_slider_start_pos[1])

        self.joint2 = tk.Scale(self.master, variable=tk.IntVar(), from_=-180, to=180)
        self.joint2["command"] = lambda x: self.set_joint_values(1, x)
        self.joint2.place(x=self.joint_slider_start_pos[0] + 100, y=self.joint_slider_start_pos[1])

        self.joint3 = tk.Scale(self.master, variable=tk.IntVar(), from_=-180, to=180)
        self.joint3["command"] = lambda x: self.set_joint_values(2, x)
        self.joint3.place(x=self.joint_slider_start_pos[0] + 150, y=self.joint_slider_start_pos[1])

        self.joint4 = tk.Scale(self.master, variable=tk.IntVar(), from_=-180, to=180)
        self.joint4["command"] = lambda x: self.set_joint_values(3, x)
        self.joint4.place(x=self.joint_slider_start_pos[0] + 200, y=self.joint_slider_start_pos[1])

        self.joint5 = tk.Scale(self.master, variable=tk.IntVar(), from_=-180, to=180)
        self.joint5["command"] = lambda x: self.set_joint_values(4, x)
        self.joint5.place(x=self.joint_slider_start_pos[0] + 250, y=self.joint_slider_start_pos[1])

        self.joint6 = tk.Scale(self.master, variable=tk.IntVar(), from_=-180, to=180)
        self.joint6["command"] = lambda x: self.set_joint_values(5, x)
        self.joint6.place(x=self.joint_slider_start_pos[0] + 300, y=self.joint_slider_start_pos[1])

        self.angle_btn = tk.Button(self.master, text="Send Joints")
        self.angle_btn["command"] = self.send_angles
        self.angle_btn.place(x=400, y=50)

        self.get_joints_btn = tk.Button(self.master, text="Get Joints")
        self.get_joints_btn["command"] = self.get_joints
        self.get_joints_btn.place(x=400, y=100)

    def set_joint_values(self, joint_idx, degree):
        self.joint_control_values[joint_idx] = int(degree)

    def send_angles(self):
        print("joint_control_values:", self.joint_control_values)
        mycobot.send_angles(self.joint_control_values, 50)

    def get_joints(self):
        print("joints_values:", mycobot.get_angles())

    def create_pos_control_widgets(self):
        self.x = tk.Scale(self.master, variable=tk.IntVar(), from_=-500, to=500)
        self.x["command"] = lambda x: self.set_pos_values(0, x)
        self.x.place(x=self.pos_slider_start_pos[0] + 50, y=self.pos_slider_start_pos[1])

        self.y = tk.Scale(self.master, variable=tk.IntVar(), from_=-500, to=500)
        self.y["command"] = lambda x: self.set_pos_values(1, x)
        self.y.place(x=self.pos_slider_start_pos[0] + 100, y=self.pos_slider_start_pos[1])

        self.z = tk.Scale(self.master, variable=tk.IntVar(), from_=-500, to=500)
        self.z["command"] = lambda x: self.set_pos_values(2, x)
        self.z.place(x=self.pos_slider_start_pos[0] + 150, y=self.pos_slider_start_pos[1])

        self.yaw = tk.Scale(self.master, variable=tk.IntVar(), from_=-500, to=500)
        self.yaw["command"] = lambda x: self.set_pos_values(3, x)
        self.yaw.place(x=self.pos_slider_start_pos[0] + 200, y=self.pos_slider_start_pos[1])

        self.pitch = tk.Scale(self.master, variable=tk.IntVar(), from_=-500, to=500)
        self.pitch["command"] = lambda x: self.set_pos_values(4, x)
        self.pitch.place(x=self.pos_slider_start_pos[0] + 250, y=self.pos_slider_start_pos[1])

        self.row = tk.Scale(self.master, variable=tk.IntVar(), from_=-500, to=500)
        self.row["command"] = lambda x: self.set_pos_values(5, x)
        self.row.place(x=self.pos_slider_start_pos[0] + 300, y=self.pos_slider_start_pos[1])

        self.coord_btn = tk.Button(self.master, text="Send Coords")
        self.coord_btn["command"] = self.send_coords
        self.coord_btn.place(x=400, y=200)

        self.get_coords_btn = tk.Button(self.master, text="Get Coords")
        self.get_coords_btn["command"] = self.get_coords
        self.get_coords_btn.place(x=400, y=250)

    def set_pos_values(self, joint_idx, pos):
        self.pos_control_values[joint_idx] = int(pos)

    def send_coords(self):
        print("pos_control_values:", self.pos_control_values)
        mycobot.send_coords(self.pos_control_values, 50, 0)

    def get_coords(self):
        print("corrds_values:", mycobot.get_coords())


if __name__ == "__main__":
    port = "/dev/ttyUSB0"
    baud = 115200
    mycobot = MyCobot(port, baud)

    root = tk.Tk()
    root.geometry("600x400")
    root.resizable(width=False, height=False)

    app = Application(master=root)
    app.master.title("MyCobot GUI")
    app.mainloop()
