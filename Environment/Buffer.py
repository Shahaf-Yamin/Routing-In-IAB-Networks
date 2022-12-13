import numpy as np
import enum
import heapq
import copy

'''
    Exceptions
'''
class InvalidMessageTypeException(Exception):
    def __init__(self):
        super().__init__('Received Invalid Message Type!')
class PacketLossOverflowException(Exception):
    def __init__(self):
        super().__init__('Lost a packet due to buffer overflow')
class PacketLossExpirationException(Exception):
    def __init__(self):
        super().__init__('Lost a packet because of TTL expiration')
'''
    Enums
'''
class message_type(enum.Enum):
    URLLC = 0
    eMBB = 1
    def __int__(self):
        if self == message_type.URLLC:
            return 0
        elif self == message_type.eMBB:
            return 1
        else:
            raise InvalidMessageTypeException()
    def __lt__(self, other):
        if self == message_type.URLLC:
            return False
        else:
            return True
    def __gt__(self, other):
        if self == message_type.URLLC:
            return True
        else:
            return False

'''
    Classes
'''
class Packet(object):
    def __init__(self, message_type, destination, node, config=None):
        self.packet_ID = np.random.randint(low=0,high=10000000, size=1)
        if message_type == message_type.URLLC:
            self.length = 1
            if config is not None:
                self.TTL = config['URLLC']['TTL']
            self.priority = message_type.URLLC
        else:
            # Handle eMBB
            # TODO: add support over here to a stream larger than one
            self.length = 1 #np.random.poisson(lam=1, size=1)[0]
            if config is not None:
                self.TTL = config['eMBB']['TTL']
            self.priority = message_type.eMBB
        if config is not None:
            self.org_TTL = self.TTL
        self.current_queue_time = 0
        self.destination = destination
        self.delay = 0
        self.iab_destination = destination.current_working_bs
        self.org_length = self.length
        self.path = None
        self.current = node
        self.history = [node]
        self._estimation = None

    def update_owner(self,owner):
        self.current = owner
        self.history.append(owner)

    def set_path(self,path):
        self.path = path  # this is the path that this packet need to go through from this step forward.

    def __len__(self):
        return self.length

    def __str__(self):
        return f'ID:{str(self.packet_ID)} to {str(self.destination)} with length {self.__len__()}'

    @property
    def estimation(self):
        return self._estimation

    @estimation.setter
    def estimation(self, val):
        self._estimation = val

    @classmethod
    def copy_ctor(cls, class_instance):
        data = Packet(class_instance.priority,class_instance.destination, class_instance.current)
        data.packet_ID = class_instance.packet_ID
        data.TTL = class_instance.TTL
        data.org_TTL = class_instance.org_TTL
        data.length = class_instance.length
        data.org_length = class_instance.org_length
        data.path = class_instance.path
        return data

class QoS_packet(Packet):
    def __init__(self, message_type, destination, node, config=None):
        super().__init__(message_type, destination, node, config)

    def __lt__(self, other):
        if other.priority == self.priority:
            return other.TTL < self.TTL
        return other.priority < self.priority

    def __le__(self, other):
        if other.priority == self.priority:
            return other.TTL <= self.TTL
        return other.priority <= self.priority

    def __ge__(self, other):
        if other.priority == self.priority:
            return other.TTL >= self.TTL
        return other.priority >= self.priority

    def __gt__(self, other):
        if other.priority == self.priority:
            return other.TTL > self.TTL
        return other.priority > self.priority

class TTL_packet(Packet):
    def __init__(self, message_type, destination, node, config=None):
        super().__init__(message_type, destination, node, config)

    def __lt__(self, other):
        return other.TTL < self.TTL

    def __le__(self, other):
        return other.TTL <= self.TTL

    def __ge__(self, other):
        return other.TTL >= self.TTL

    def __gt__(self, other):
        return other.TTL > self.TTL

class PriorityQueue:
    def __init__(self, max_size=0, is_fifo=False):
        self._data = []
        self._index = 0
        self.max_buffer_size = max_size
        self.is_fifo = is_fifo

    def append(self, item):
        if self._index > self.max_buffer_size:
            return item
        #TODO: handle the scenario of excedding the size of the queue!!!
        if self.is_fifo is False:
            heapq.heappush(self._data, item)
        else:
            self._data.append(item)
        self._index += 1

    def isEmpty(self):
        return len(self._data) == 0

    def isFull(self):
        return len(self._data) == self.max_buffer_size

    def pop(self):
        if self.is_fifo is False:
            item = heapq.heappop(self._data)
        else:
            item = self._data.pop(0)
        self._index -= 1
        return item.get_index()

    def remove(self, RemoveIndex):
        for idx, data in enumerate(self._data):
            if data.get_index() == RemoveIndex:
                data = self._data.pop(idx)
                return
        raise Exception('Tried to remove a packet index which is not in the current buffer!')

    def index(self, item):
        ordered = []
        temp = self._data[:]
        while temp:
            ordered.append(heapq.heappop(temp))
        return ordered.index(item)

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __add__(self, other):
        newQueue = PriorityQueue(self.max_buffer_size,self.is_fifo)
        newQueue._data = self._data
        for elem in other:
            newQueue.append(elem)
        return newQueue
'''
Functions
'''
def findMaximumElement(heap, n):
    '''
    Function to find the maximum
    element in a minimum heap
    :param heap: required heap
    :return: maximum element
    '''
    maximumElement = heap[n // 2]

    for i in range(1 + n // 2, n):
        maximumElement = max(maximumElement, heap[i])
    return maximumElement