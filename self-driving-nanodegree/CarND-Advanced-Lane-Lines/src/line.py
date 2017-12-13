from collections import deque

# Define a class to receive the characteristics of each line detection
class Line():
    def add_to_queue(self, m_queue, m_element):
        if (len(m_queue) >= self.n):
            m_queue.popleft()
        m_queue.append(m_element)

    def __init__(self):
        # maximum n
        self.n = 5
        # list for coefficients
        self.leftx_hist = deque(maxlen=5)
        self.lefty_hist = deque(maxlen=5)
        self.rightx_hist = deque(maxlen=5)

        self.curr_leftx = None
        self.curr_lefty = None
        self.curr_rightx = None


    def add_iteration(self, leftx, lefty, rightx):
        # add to the end of a list 
        self.add_to_queue(self.leftx_hist, leftx)
        self.add_to_queue(self.lefty_hist, lefty)
        self.add_to_queue(self.rightx_hist, rightx)

    def get_average(self, queue):
        m_sum = queue[0]
        for i in range(1, len(queue)):
            m_sum += queue[i]
        return m_sum/len(queue)

    def get_smoothed_line(self):
        
        self.curr_leftx = self.get_average(self.leftx_hist)
        self.curr_lefty = self.get_average(self.lefty_hist)
        self.curr_rightx = self.get_average(self.rightx_hist)

        return self.curr_leftx, self.curr_lefty, self.curr_rightx
        
        