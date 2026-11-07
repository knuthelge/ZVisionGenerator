<script lang="ts">
  import Modal from './Modal.svelte';
  import Select from './Select.svelte';

  const OPTIONS = [
    { value: 'a', label: 'A' },
    { value: 'b', label: 'B' },
  ];
</script>

<Modal open title="Pick one">
  <Select id="choice" options={OPTIONS} value="a" />
</Modal>
